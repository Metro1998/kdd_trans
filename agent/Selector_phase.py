# _*_ coding：utf-8_*_
# 开发人员：zm
# 开发时间：17:42
# 文件名：Selector_phase.py
# 开发工具：PyCharm
from keras.engine.topology import Layer
import keras.backend as K


class Selector(Layer):

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
