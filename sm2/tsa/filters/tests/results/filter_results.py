import numpy as np

conv2 = np.array([118.75, 155.5, 197.75, 238.75, 239.5, 228, 303, 629,
                  789, 599.75, 499.25, 527.25, 477.75, 260.5, 180.75,
                  296.75, 263.5, 155.75, 108.5, 199.5, 379.75, 109.5,
                  74.25, 56.75, -95, -93, -29, 90.5, 302, 56.5, 126.75,
                  np.nan])

conv1 = np.array([np.nan, 118.75, 155.5, 197.75, 238.75, 239.5, 228,
                  303, 629, 789, 599.75, 499.25, 527.25, 477.75, 260.5,
                  180.75, 296.75, 263.5, 155.75, 108.5, 199.5, 379.75,
                  109.5, 74.25, 56.75, -95, -93, -29, 90.5, 302, 56.5,
                  126.75])

recurse = np.array([-50, 137.5, 239.625, 428.09375, 627.9765625,
                    815.005859375, 993.248535156, 1277.68786621,
                    1935.57803345, 2580.10549164, 2948.97362709,
                    3345.75659323, 3786.56085169, 4133.35978708,
                    4241.66005323, 4390.58498669, 4690.35375333,
                    4854.41156167, 4941.39710958, 5021.6507226,
                    5233.58731935, 5609.60317016, 5518.59920746,
                    5639.35019814, 5652.16245047, 5507.95938738,
                    5467.01015315, 5464.24746171, 5589.93813457,
                    5919.51546636, 5792.12113341, 6007.96971665])

recurse_init = np.array([87.5, 278.125, 379.46875, 568.1328125,
                         767.966796875, 955.008300781, 1133.2479248,
                         1417.6880188, 2075.5779953, 2720.10550117,
                         3088.97362471, 3485.75659382, 3926.56085154,
                         4273.35978711, 4381.66005322, 4530.58498669,
                         4830.35375333, 4994.41156167, 5081.39710958,
                         5161.6507226, 5373.58731935, 5749.60317016,
                         5658.59920746, 5779.35019814, 5792.16245047,
                         5647.95938738, 5607.01015315, 5604.24746171,
                         5729.93813457, 6059.51546636, 5932.12113341,
                         6147.96971665])

conv2_na = np.array([118.75, 155.5, 197.75, 238.75, 239.5, 228, 303,
                     629, np.nan, np.nan, 499.25, 527.25, 477.75,
                     260.5, 180.75, 296.75, 263.5, 155.75, 108.5,
                     199.5, 379.75, 109.5, 74.25, 56.75, -95, -93,
                     -29, 90.5, 302, 56.5, 126.75, np.nan])

conv1_na = np.array([np.nan, 118.75, 155.5, 197.75, 238.75, 239.5,
                     228, 303, 629, np.nan, np.nan, 499.25, 527.25,
                     477.75, 260.5, 180.75, 296.75, 263.5, 155.75,
                     108.5, 199.5, 379.75, 109.5, 74.25, 56.75, -95,
                     -93, -29, 90.5, 302, 56.5, 126.75])

recurse_na = np.array([-50, 137.5, 239.625, 428.09375, 627.9765625,
                       815.005859375, 993.248535156, 1277.68786621,
                       1935.57803345, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                       np.nan])

recurse_init_na = np.array([87.5, 278.125, 379.46875, 568.1328125,
                            767.966796875, 955.008300781, 1133.2479248,
                            1417.6880188, 2075.5779953, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan])

conv2_odd = np.array([np.nan, np.nan, 366.95, 412.75, 419.05, 501.15,
                      850.85, 1138.65, 1109, 1053.15, 1043.2, 946.35,
                      687.55, 523.5, 544.65, 485.25, 371.3, 297.2,
                      344.9, 517.85, 319.55, 260.3, 191.15, -11.35,
                      -95.45, -72.15, 40.25, 299.85, 173.95, 247.5,
                      np.nan, np.nan])

conv1_odd = np.array([np.nan, np.nan, np.nan, np.nan, 366.95, 412.75,
                      419.05, 501.15, 850.85, 1138.65, 1109, 1053.15,
                      1043.2, 946.35, 687.55, 523.5, 544.65, 485.25,
                      371.3, 297.2, 344.9, 517.85, 319.55, 260.3,
                      191.15, -11.35, -95.45, -72.15, 40.25, 299.85,
                      173.95, 247.5])

recurse_odd = np.array([191.5, 462.125, 668.84375, 1044.1453125,
                        1556.46835938, 2238.65205078, 3175.44806152,
                        4572.56601685, 6833.45236176, 9776.38394429,
                        13509.7387615, 18791.5897613, 26145.4239591,
                        36153.4065035, 49699.8299323, 68480.4947171,
                        94501.551723, 130110.583827, 179061.168784,
                        246469.715955, 339396.406323, 467401.785292,
                        643016.749056, 885080.436404, 1218108.49028,
                        1676305.60832, 2307074.11064, 3175195.69641,
                        4370080.25182, 6014713.24095, 8277634.14851,
                        11392536.8578])