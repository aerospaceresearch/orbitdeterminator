## Orbit determination

### Coding Guidelines
We are folloing standard PEP8 guidelines
For docstrings, we are following [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google)

* read_data.py

Read data from a folder in the same directory as that of read_data.py

Usage : python3 read_data.py [folder_name]


* triple_moving_average.py

apply custom filter to the jittery data. The filter give mse value of `22.6214472424` compared to orbit0jittery.csv which gives mse value of `47.8632703865`

Usage: python3 triple_moving_average.py filename.csv

* mse.py

Calculate the mean error in the true values and filtered values(take orbit0perfect.csv as reference for true orbit)

Usage: python3 mse.py true.csv test.csv

