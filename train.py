#!/usr/bin/env python

import keras, models, losses

m = models.getModel("CascadeNet-5", (8,9,1))
m.summary()
