import tensorflow as tf
import math

class HellingerDistanceThreeClass(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HellingerDistanceThreeClass, self).__init__(**kwargs)

    def compute_belief_hellinger_distance(self, m1Road, m1Vehicle, m1Background, m1Ignorance, m2, cardRoad=1, cardVehicle=1, cardBackground=1, cardIgnorance=3):
        """
        Computes the Belief Hellinger distance for each class given the input belief masses.
        """
        # Unpack the m2 values (representing singleton hypotheses for comparison)
        m2Road, m2Vehicle, m2Background, m2Ignorance = m2

        # Calculate the scaled terms based on cardinalities
        sRoad = (tf.math.sqrt(m1Road) - tf.math.sqrt(m2Road)) ** 2 / (2**cardRoad - 1)
        sVehicle = (tf.math.sqrt(m1Vehicle) - tf.math.sqrt(m2Vehicle)) ** 2 / (2**cardVehicle - 1)
        sBackground = (tf.math.sqrt(m1Background) - tf.math.sqrt(m2Background)) ** 2 / (2**cardBackground - 1)
        sIgnorance = (tf.math.sqrt(m1Ignorance) - tf.math.sqrt(m2Ignorance)) ** 2 / (2**cardIgnorance - 1)

        # Calculate the Belief Hellinger distance
        bh_distance = (1 / math.sqrt(2)) * tf.math.sqrt(sRoad + sVehicle + sBackground + sIgnorance)
        return bh_distance

    def call(self, x):
        # Unpack the m-values from input tensor x
        m1pRoad = x[:, :, :, 0]       # m(Road)
        m1pVehicle = x[:, :, :, 1]    # m(Vehicle)
        m1pBackground = x[:, :, :, 2] # m(Background)
        m1pIgnorance = x[:, :, :, 3]  # m(Road ∪ Vehicle ∪ Background)

        # Compute Hellinger distances for each class hypothesis
        bh_m1Road = self.compute_belief_hellinger_distance(m1pRoad, m1pVehicle, m1pBackground, m1pIgnorance, [1.0, 0.0, 0.0, 0.0])  # for Road
        bh_m1Vehicle = self.compute_belief_hellinger_distance(m1pRoad, m1pVehicle, m1pBackground, m1pIgnorance, [0.0, 1.0, 0.0, 0.0])  # for Vehicle
        bh_m1Background = self.compute_belief_hellinger_distance(m1pRoad, m1pVehicle, m1pBackground, m1pIgnorance, [0.0, 0.0, 1.0, 0.0])  # for Background
        bh_m1Ignorance = self.compute_belief_hellinger_distance(m1pRoad, m1pVehicle, m1pBackground, m1pIgnorance, [0.0, 0.0, 0.0, 1.0])  # for Ignorance

        # Determine decision based on the smallest Hellinger distance
        decision_road = tf.math.less(bh_m1Road, tf.minimum(tf.minimum(bh_m1Vehicle, bh_m1Background), bh_m1Ignorance))
        decision_vehicle = tf.math.less(bh_m1Vehicle, tf.minimum(tf.minimum(bh_m1Road, bh_m1Background), bh_m1Ignorance))
        decision_background = tf.math.less(bh_m1Background, tf.minimum(tf.minimum(bh_m1Road, bh_m1Vehicle), bh_m1Ignorance))
        decision_ignorance = tf.math.less(bh_m1Ignorance, tf.minimum(tf.minimum(bh_m1Road, bh_m1Vehicle), bh_m1Background))

        # Expand dimensions for concatenation
        decision_road = tf.expand_dims(decision_road, axis=-1)
        decision_vehicle = tf.expand_dims(decision_vehicle, axis=-1)
        decision_background = tf.expand_dims(decision_background, axis=-1)
        decision_ignorance = tf.expand_dims(decision_ignorance, axis=-1)

        # Concatenate decisions into a single output tensor
        decision_out = tf.concat([decision_road, decision_vehicle, decision_background, decision_ignorance], axis=-1)

        # Cast output to uint8 for further processing
        decision_out = tf.cast(decision_out, tf.uint8)

        return decision_out

    def get_config(self):
        config = super(HellingerDistanceThreeClass, self).get_config()
        return config
