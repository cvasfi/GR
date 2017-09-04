import tensorflow as tf
import numpy as np
import six
import sys
import time
import input_pipeline
import AlexNet_TI


_extra_train_ops = []

def train(batch_size,classes,FLAGS):
    print("train starts")
    filenames, all_labels = input_pipeline.read_labeled_image_list("training_labels.csv")
    images, labels = input_pipeline.load_batches(image_filenames=filenames,
                 label_filenames=all_labels,
                 network="AlexNet",
                 shape=(60, 60, 3),
                 batch_size=100)
    print("labels shale")
    print(labels.get_shape)


    model = AlexNet_TI.model(classes,"train",batch_size)


    model.build(images,labels)

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

    train_ops = [apply_op]
    train_op = tf.group(*train_ops)


    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir="alexNet_model/train",
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
            self._lrn_rate=0.01/(10**(train_step/5000))


    with tf.train.MonitoredTrainingSession(
            checkpoint_dir="alexNet_model",
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

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('cnn', 'normal', 'normal or dws')
tf.app.flags.DEFINE_string('mode', 'train', 'train or test.')

def main(_):
    dev = '/gpu:0'

    print("start program")
    with tf.device(dev):
        if FLAGS.mode=="train":
            train(100,10,FLAGS)
        else:
            eval(1,10,FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()