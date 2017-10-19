import tensorflow as tf
import numpy as np
import six
import sys
import time
import input_pipeline
import AlexNet_TI
import ResNet50_TI
from LookupConvolution2d import extract_dense_weights
from tensorflow.python.client import timeline
import logging

test_size=10000

def train(batch_size,classes,FLAGS):
    print("train starts")
    filenames, all_labels = input_pipeline.read_labeled_image_list("training_labels.csv")
    images, labels = input_pipeline.load_batches(image_filenames=filenames,
                 label_filenames=all_labels,
                 network=FLAGS.network,
                 shape=(64, 64, 3),
                 batch_size=batch_size)
    print("labels shale")
    print(labels.get_shape)


    if(FLAGS.network=="alexnet"):
        model = AlexNet_TI.model(classes,"train",batch_size)
        model_path="alexNet_model"
    elif(FLAGS.network=="resnet50"):
        model= ResNet50_TI.model(classes,"train",batch_size)
        model_path="resnet50_model"

    if(FLAGS.cnn=="dws"):
        model.build_dw(images, labels)
        model_path=model_path+"_dw"
    elif(FLAGS.cnn=="lookup"):
        model.build_lookup(images, labels)
        model_path=model_path+"_l"
    else:
        model.build(images, labels)

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    truth = tf.argmax(labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    summaries = tf.summary.merge_all()
    global_step = tf.contrib.framework.get_or_create_global_step()
    lrn_rate=   tf.constant(0.1, tf.float32)
    optimizer = tf.train.MomentumOptimizer(lrn_rate, 0.9)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(model.cost, trainable_variables)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=tf.contrib.framework.get_or_create_global_step(), name='train_step')

    train_ops = [apply_op]+ model._extra_train_ops
    train_op = tf.group(*train_ops)


    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=(model_path+"/train"),
        summary_op=tf.summary.merge([summaries,
                                     tf.summary.scalar('Precision', precision)]))

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=100)

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = 0.01

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                global_step,  # Asks for global step value.
                feed_dict={lrn_rate: self._lrn_rate})  # Sets learning rate

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            self._lrn_rate=0.01/(10**(train_step/40000))


    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=model_path,
            hooks=[logging_hook, _LearningRateSetterHook()],
            chief_only_hooks=[summary_hook],
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        while not mon_sess.should_stop():
            #a, a_q=mon_sess.run([model.sample,model.sample_quantized])
            #print(a[0,0,0,:])
            #print(a_q[0,0,0,:])
            #print("___")
            mon_sess.run(train_op)

def eval(batch_size,classes,FLAGS):
    print("eval starts")
    filenames, all_labels = input_pipeline.read_labeled_image_list("validation_labels.csv")
    images, labels = input_pipeline.load_batches(image_filenames=filenames,
                 label_filenames=all_labels,
                 network=FLAGS.network,
                 shape=(64, 64, 3),
                 batch_size=batch_size)
    print("labels shale")
    print(labels.get_shape)

    if(FLAGS.network=="alexnet"):
        model = AlexNet_TI.model(classes,"test",batch_size)
        model_path="alexNet_model"
    elif(FLAGS.network=="resnet50"):
        model= ResNet50_TI.model(classes,"test",batch_size)
        model_path="resnet50_model"

    if(FLAGS.cnn=="dws"):
        model.build_dw(images, labels)
        model_path=model_path+"_dw"
    elif (FLAGS.cnn == "lookup"):
        model.build_lookup(images, labels)
        model_path = model_path + "_l"
    else:
        model.build(images, labels)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(model_path+"/test")

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)
    best_precision = 0.0

    i=0
    total_duration=0

    try:
        ckpt_state = tf.train.get_checkpoint_state(model_path)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', model_path)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    model_summaries = tf.summary.merge_all()
    total_prediction, correct_prediction = 0, 0
    duration=0

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for counter in six.moves.range(test_size/batch_size):
        start_time = time.time()
        (summaries, loss, predictions, truth, train_step) = sess.run(
            [model_summaries, model.cost, model.predictions,
             labels, model.global_step], options=run_options, run_metadata=run_metadata)
        if(counter != 0):
            duration += time.time()-start_time
        # print("duration is: "+str(duration))
        # print(duration)
    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))
    summary_writer.flush()

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(model_path+'_timeline.json', 'w') as f:
        f.write(ctf)

    print"average duration of a single image:" + str((duration/((test_size/batch_size)-1))/batch_size)
    print"duration of a single batch:" + str((duration/((test_size/batch_size)-1)))
    print"batches taken into account:" + str((((test_size/batch_size)-1)))
    print"total:" + str((duration))

####################################################

def lookup_eval(batch_size,classes,FLAGS):

    print("lookup initialization starts")
    filenames, all_labels = input_pipeline.read_labeled_image_list("validation_labels.csv")
    images, labels = input_pipeline.load_batches(image_filenames=filenames,
                                                 label_filenames=all_labels,
                                                 network=FLAGS.network,
                                                 shape=(64, 64, 3),
                                                 batch_size=batch_size)
    print("labels shale")
    print(labels.get_shape)

    if (FLAGS.network == "alexnet"):
        model = AlexNet_TI.model(classes, "test", batch_size)
        model_path = "alexNet_model"
    elif (FLAGS.network == "resnet50"):
        model = ResNet50_TI.model(classes, "test", batch_size)
        model_path = "resnet50_model"

    if (FLAGS.cnn == "dws"):
        model.build_dw(images, labels)
        model_path = model_path + "_dw"
    elif (FLAGS.cnn == "lookup"):
        model.build_lookup(images, labels)
        model_path = model_path + "_l"
    else:
        model.build(images, labels)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(model_path + "/test")

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)
    best_precision = 0.0

    i = 0
    total_duration = 0

    try:
        ckpt_state = tf.train.get_checkpoint_state(model_path)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', model_path)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    model_summaries = tf.summary.merge_all()
    total_prediction, correct_prediction = 0, 0

    (summaries, loss, predictions, truth, train_step) = sess.run(
        [model_summaries, model.cost, model.predictions,
         labels, model.global_step])
    if (FLAGS.network == "alexnet"):
        gr = tf.get_default_graph()
        tensors = [gr.get_tensor_by_name('C%d/align_conv/kernel:0' % (convid + 1)) for convid in range(5)]
        kernel_vals = sess.run(tensors)
        for kernel_val in kernel_vals:
            print kernel_val.size
            print (np.count_nonzero(kernel_val) / kernel_val.size)

    extract_dense_weights(sess)

    tf.reset_default_graph()

    ##############################################################

    print("lookup eval starts")
    filenames, all_labels = input_pipeline.read_labeled_image_list("validation_labels.csv")
    images, labels = input_pipeline.load_batches(image_filenames=filenames,
                 label_filenames=all_labels,
                 network=FLAGS.network,
                 shape=(64, 64, 3),
                 batch_size=batch_size)
    print("labels shale")
    print(labels.get_shape)

    if(FLAGS.network=="alexnet"):
        model = AlexNet_TI.model(classes,"test",batch_size)
        model_path="alexNet_model"
    elif(FLAGS.network=="resnet50"):
        model= ResNet50_TI.model(classes,"test",batch_size)
        model_path="resnet50_model"

    if(FLAGS.cnn=="dws"):
        model.build_dw(images, labels)
        model_path=model_path+"_dw"
    elif (FLAGS.cnn == "lookup"):
        model.build_lookup(images, labels)
        model_path = model_path + "_l"
    else:
        model.build(images, labels)

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(model_path+"/test")

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)
    best_precision = 0.0

    i=0
    total_duration=0

    try:
        ckpt_state = tf.train.get_checkpoint_state(model_path)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', model_path)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    model_summaries = tf.summary.merge_all()
    total_prediction, correct_prediction = 0, 0
    duration=0

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for counter in six.moves.range(test_size/batch_size):
        start_time = time.time()
        (summaries, loss, predictions, truth, train_step) = sess.run(
            [model_summaries, model.cost, model.predictions,
             labels, model.global_step], options=run_options, run_metadata=run_metadata)
        if(counter != 0):
            duration += time.time()-start_time
        print("duration is: "+str(duration))
        print(duration)
        truth = np.argmax(truth, axis=1)
        predictions = np.argmax(predictions, axis=1)
        correct_prediction += np.sum(truth == predictions)
        total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))
    summary_writer.flush()


    print"average duration of a single image:" + str((duration/((test_size/batch_size)-1))/batch_size)
    print"duration of a single batch:" + str((duration/((test_size/batch_size)-1)))
    print"batches taken into account:" + str((((test_size/batch_size)-1)))
    print"total:" + str((duration))
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(model_path+'_timeline.json', 'w') as f:
        f.write(ctf)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cnn', 'normal', 'normal or dws')
tf.app.flags.DEFINE_string('mode', 'train', 'train or test.')
tf.app.flags.DEFINE_string('network', 'alexnet', 'alexnet or resnet.')

def main(_):
    dev = '/gpu:0'

    print("start program")
    with tf.device(dev):
        if FLAGS.mode=="train":
            train(50,200,FLAGS)
        else:
            if (FLAGS.cnn=="lookup"):
                lookup_eval(1,200,FLAGS)
            else:
                eval(1,200,FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()