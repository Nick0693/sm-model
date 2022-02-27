# RNN-SM-Model
Recurrent Neural Network model for remote sensing-based soil moisture estimation + data pipeline

## Setup

### Installing a python environment
Miniconda is a minimal installer for conda and the simplest way to get started with python. The installer can be found [here](https://docs.conda.io/en/latest/miniconda.html)

#### Adding Miniconda to the system variables
  * Once installed, open the Miniconda command prompt and type `rundll32 sysdm.cpl,EditEnvironmentVariables`
  * Double-click the 'Path' variable under the 'User' section
  * Click the 'Edit text' button in the 'Edit' dialog window and add the following string to the existing 'Path' variable value `%UserProfile%\miniconda3\condabin;`
  * In the Miniconda command prompt, type the following to append the path `setx Path "%Path%%UserProfile%\miniconda3\condabin;"`

### Setting up an environment with dependencies
Environments allow you keep different versions of packages for separate projects. Install the environment using the **environment.yml** file and then activate the environment in the Miniconda prompt with `conda activate soil-moisture-rnn`.

### Setting up the GEE configurations
The script relies on the Google Earth Engine python API to extract geospatial information from various layers. To utilize this data the end user must first authorize their account via Google. The authentication is a one-time process that will write a credential file to your local directory with a token. This token is written by typing `earthengine authenticate` in the Miniconda prompt.


## Preprocessing data

The preprocessing workflow currently works with a collection of ICOS stations ([download from here](https://www.icos-cp.eu/data-services/about-data-portal)) or a collection of sensors from a single network from ISMN ([download from here](https://ismn.geo.tuwien.ac.at/en/)).


All functions rely on the configurations found in the *settings.yml* file and uses the path to this file as the only argument.


ICOS and ISMN data can be preprocessed via the *reader* functions accordingly:
```
icos_reader.initialize(/settings.yml)
ismn_reader.initialize(/settings.yml)
```

The *reader* functions will output a .csv and .yml file in the working directory (wrk_dir in *settings.yml*). The .yml is a site_info file that contains basic information about each measuring station. The .csv file is a harmonised dataframe of the variables collected at the stations. **Do not alter the naming** of this output as it is later read by the compiler.


The ICOS and ISMN data can be merged with remotely sensed variables retrieved from the Google Earth Engine through the *preprocessing* function:
```
preprocessing.compile_data(/settings.yml)
```

The output is a .csv file containing information from all specified variables at daily intervals.


## Running the model

*Under construction*
