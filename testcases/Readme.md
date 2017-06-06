## Orbit determination


* read_data.py

Read data from a folder in the same directory as that of read_data.py
Usage : python3 read_data.py [folder_name]



* lamberts.py & orbit_fit.py

For these two scripts lamberts and orbit_fit python 3.4 is need, because the PyKEP library only works in python 3.4 version and
not python 3.6

* triple_moving_average.py

apply custom filter to the jittery data. The filter give mse value of `22.6214472424` compared to orbit0jittery.csv which gives mse value of `47.8632703865`
Usage: python3 triple_moving_average.py filename.csv

* mse.py

Calculate the mean error in the true values and filtered values(take orbit0perfect.csv as reference for true orbit)
Usage: python3 mse.py true.csv test.csv

