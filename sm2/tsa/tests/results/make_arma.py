import numpy as np
import pandas as pd

from sm2.tsa.arima_process import arma_generate_sample

np.random.seed(12345)

# no constant
y_arma11 = arma_generate_sample([1., -.75],
                                [1., .35],
                                nsample=250)
y_arma14 = arma_generate_sample([1., -.75],
                                [1., .35, -.75, .1, .35],
                                nsample=250)
y_arma41 = arma_generate_sample([1., -.75, .25, .25, -.75],
                                [1., .35],
                                nsample=250)
y_arma22 = arma_generate_sample([1., -.75, .45],
                                [1., .35, -.9],
                                nsample=250)
y_arma50 = arma_generate_sample([1., -.75, .35, -.3, -.2, .1],
                                [1.],
                                nsample=250)
y_arma02 = arma_generate_sample([1.],
                                [1., .35, -.75],
                                nsample=250)


# constant
constant = 4.5
y_arma11c = constant + arma_generate_sample([1., -.75],
                                            [1., .35],
                                            nsample=250)
y_arma14c = constant + arma_generate_sample([1., -.75],
                                            [1., .35, -.75, .1, .35],
                                            nsample=250)
y_arma41c = constant + arma_generate_sample([1., -.75, .25, .25, -.75],
                                            [1., .35],
                                            nsample=250)
y_arma22c = constant + arma_generate_sample([1., -.75, .45],
                                            [1., .35, -.9],
                                            nsample=250)
y_arma50c = constant + arma_generate_sample([1., -.75, .35, -.3, -.2, .1],
                                            [1.],
                                            nsample=250)
y_arma02c = constant + arma_generate_sample([1.], [1., .35, -.75],
                                            nsample=250)


data = np.column_stack((y_arma11, y_arma14, y_arma41,
                        y_arma22, y_arma50, y_arma02,
                        y_arma11c, y_arma14c, y_arma41c,
                        y_arma22c, y_arma50c, y_arma02c))

df = pd.DataFrame(data,
                  columns=['y_arma11', 'y_arma14', 'y_arma41',
                           'y_arma22', 'y_arma50', 'y_arma02',
                           'y_arma11c', 'y_arma14c', 'y_arma41c',
                           'y_arma22c', 'y_arma50c', 'y_arma02c'])
df.to_csv('y_arma_data.csv', index=False)
