from potentials import SIEP_plus_XS
from math import log10
import re

# (b, e, te, g, tg, s, xs, ys) = (1.0, 0.0, 0.0, 0.27, 0.0, 0.0, 0.04, 0.1)

(b, e, te, g, tg, s, xs, ys) = (1.0, 0.3, 90.0, 0.0, 0.0, 0.0, 0.04, 0.1)

pot_params = {        "b": b,
                    "x_g": 0,
                    "y_g": 0,
                    "eps": e,
              "eps_theta": te - 90,
                  "gamma": g,
            "gamma_theta": tg - 90}
potential = SIEP_plus_XS(**pot_params)
img = potential.get_image_configuration(x_s=xs, y_s=ys)
nimg = 0
for k in img.keys():
  if(re.match("^x_[0-9]+$", k) != None):
    nimg = nimg + 1
print("%23s %23s %23s %23s" % ("x", "y", "mag", "mu"))
for i in range(nimg):
  x = img[f'x_{i+1}']
  y = img[f'y_{i+1}']
  mag = img[f'mu_{i+1}']
  mu = -2.5 * log10(abs(mag)) # I am not sure what mu means
  print("%+23.16e %+23.16e %+23.16e %+23.16e" % (x,y,mag,mu))
