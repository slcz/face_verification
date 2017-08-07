import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
import random
from sklearn.svm import SVC
import align.detect_face
from scipy import misc

config = tf.ConfigProto( device_count = {'GPU': 0})

def main(args):
    with tf.Graph().as_default():
        with tf.Session(config = config) as sess:
            np.random.seed(seed=args.seed)

            labels = []
            paths  = []
            dirs = os.listdir(args.data_dir)
            for i in dirs:
                path = os.path.join(args.data_dir, i)
                if not os.path.isdir(path):
                    continue
                files = os.listdir(path)
                for f in files:
                    if i == args.name:
                        labels.append(True)
                    else:
                        labels.append(False)
                    paths.append(os.path.join(path, f))

            length = len(labels)
            x = np.arange(length)
            np.random.shuffle(x)
            labels = [labels[i] for i in x]
            paths  = [paths [i] for i in x]
            length = args.train_size
            if args.mode == 'TRAIN':
                labels = labels[:length]
                paths  = paths [:length]
            else:
                labels = labels[length:]
                paths  = paths [length:]

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = load_and_align_data(paths_batch, args.image_size, 44)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            classifier_filename = args.name

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Saving classifier model
                with open(classifier_filename, 'wb') as outfile:
                    pickle.dump((model, [False, True]), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename)

            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file {}'.format(classifier_filename))

                predictions = model.predict_proba(emb_array)
                predictions = predictions[:,1]
                prob_index  = np.argsort(predictions)
                for p in prob_index:
                    print("{} {:.3f}".format(labels[p], predictions[p]))

def load_and_align_data(paths, size, margin):
    with tf.Graph().as_default():
        sess = tf.Session(config = config)
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    nsamples = len(paths)
    img_list = [None] * nsamples
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    for i in range(nsamples):
        img = misc.imread(os.path.expanduser(paths[i]))
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:,:,0:3]
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, 20, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (size, size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('name',
        help='name of the person to classifiy')
    parser.add_argument('train_size', type=int,
        help = 'Number of imaged used to train', default = 10)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
