import tensorflow as tf
import numpy as np
from scipy.stats import norm

qs = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 256) # Quantiles for normal distn
weights = np.array(norm.pdf(qs)) # Probabilities of normal distn
weights = weights / weights.sum() # normalize the probabilities so their sum is 1
weights = tf.constant(weights)


@tf.function
def tf_arrange_x_sequence(m, smin=0, smax=256):
    """Arranges a looped sequence around the input value"""
    seq = tf.range(smin, smax)
    seq = tf.concat([[tf.range(smin, smax)[::-1]],
                    tf.concat([[seq], [tf.range(smin, smax - 1)[::-1]]],
                              axis=1)],
                    axis=1)
    start = 256 + m - tf.cast(tf.math.round(smax/2), tf.int32)
    end = 256 + m + tf.cast(tf.math.round(smax/2), tf.int32)
    return seq[0, start:end]

def mutate_x(x, mutation_rate, method="uniform"):
  """Add mutations to input.
  Generate mutations for all positions,
  in order to be different than itselves, the
  mutations have to be >= 1
  mute the untargeted positions by multiple mask (1 for targeted)
  then add the mutations to the original, mod 255 if necessary.
  Args:
    x: input image tensor of size batch*width*height*channel
    mutation_rate: mutation rate
    method: Method to distort images. One of ["uniform", "gaussian"]
  Returns:
    mutated input
  """
  w, h, c = x.get_shape().as_list()
  mask = tf.cast(
      tf.compat.v1.multinomial(
          tf.compat.v1.log([[1.0 - mutation_rate, mutation_rate]]), w * h * c),
      tf.int32)[0]
  mask = tf.reshape(mask, [w, h, c])

  if method == "uniform":
      possible_mutations = tf.compat.v1.random_uniform(
          [w * h * c],
          minval=0,
          maxval=256,  # 256 values [0, 1, ..., 256) = [0, 1, ..., 255]
          dtype=tf.int32)
      possible_mutations = tf.reshape(possible_mutations, [w, h, c])
      x = tf.compat.v1.mod(tf.cast(x, tf.int32) + mask * possible_mutations, 256)
      x = tf.cast(x, tf.float32)

  elif method == "gaussian":
      # Add 'gaussian' integer noise around current value of pixel
      weights = ps

      # Get a flat version of input image
      x_flat = tf.reshape(
          x[tf.cast(mask, tf.bool)],
          [tf.math.reduce_sum(mask)]
      )

      # Get the [0...255] sequences in the relevant order for each of the
      # pixels that are to be manipulated
      seqs = tf.map_fn(fn=lambda t: tf_arrange_x_sequence(t), elems=x_flat)

      # Get new values for those pixels
      # First, sample list indices
      samples = tf.map_fn(
          fn=lambda t: tf.compat.v1.multinomial(tf.compat.v1.log([weights]),1, output_dtype=tf.int32),
          elems=tf.range(x_flat.get_shape().as_list()[0])
      )
      samples = tf.reshape(samples, x_flat.get_shape().as_list())

      # Second, get values of those list indices
      mutations = tf.map_fn(
          fn=lambda t: (t[0][t[1]], tf.constant([0])),
          elems=(seqs, samples),
          dtype=(tf.int32, tf.int32)
      )[0]

      # Get indices of mask
      idx = tf.where(
          tf.cast(
              tf.reshape(mask, [w*h*c]), tf.bool
          )
      )
      idx = tf.cast(idx, tf.int32)

      # Make the mutation matrix (zeros except at mask indices)
      mutations = tf.reshape(tf.scatter_nd(idx, mutations, [w*h*c]), x.get_shape())

      x = tf.compat.v1.mod(tf.cast(x, tf.int32) + mutations,
                           256)
      x = tf.cast(x, tf.float32)

  return x


def image_preprocess_add_noise(x, mutation_rate, method):
  """Image preprocess and add noise to image."""
  x['image'] = tf.cast(x['image'], tf.float32)

  if mutation_rate > 0:
    x['image'] = mutate_x(x['image'], mutation_rate, method)

  return x  # (input, output) of the model