import geometry_msgs.msg
import rospy
import tf
import tf2_ros


class TransformListener(object):

    """ROS TF Listener class

    """

    def __init__(self, use_tf2=True):
        if use_tf2:
            try:
                self.tf_listener = tf2_ros.BufferClient("/tf2_buffer_server")
                ok = self.tf_listener.wait_for_server(rospy.Duration(10))
                if not ok:
                    raise Exception(
                        "timed out: wait_for_server for 10.0 seconds")
            except Exception as e:
                rospy.logerr("Failed to initialize tf2 client: %s" % str(e))
                rospy.logwarn("Fallback to tf client")
                use_tf2 = False
        if not use_tf2:
            self.tf_listener = tf.TransformListener()
        self.use_tf2 = use_tf2

    def _wait_for_transform_tf1(self,
                                target_frame, source_frame,
                                time, timeout):
        pass

    def _wait_for_transform_tf2(self,
                                target_frame, source_frame,
                                time, timeout):
        pass

    def wait_for_transform(self,
                           target_frame, source_frame,
                           time, timeout=rospy.Duration(0)):
        pass

    def _lookup_transform_tf1(self, target_frame, source_frame, time, timeout):
        pass

    def _lookup_transform_tf2(self, target_frame, source_frame, time, timeout):
        pass

    def lookup_transform(self,
                         target_frame,
                         source_frame,
                         time=rospy.Time(0),
                         timeout=rospy.Duration(0)):
        pass
