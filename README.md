# The economic value of neighborhoods: Predicting real estate prices from the urban environment

This repository contains all the code required to reproduce the results presented in the following paper:

* M. De Nadai, B. Lepri. *The economic value of neighborhoods: Predicting real estate prices from the urban environment*, 2018.

Input, intermediary and source data can be downloaded from [figshare](https://doi.org/10.6084/m9.figshare.6934970).

# Dependencies

Dependencies are listed in the `requirements.txt` file at the root of the repository. Using [Python 3.6](https://www.python.org/downloads/) with [pip](https://pip.pypa.io/en/stable/installing/) all the required dependencies can be installed automatically.

``` sh
pip3 install -r requirements.txt
```

* [PostgreSQL 10.0](https://www.postgresql.org/) 
* [PostGIS 2.4.1](https://postgis.net) extension

# Data

Due to storage constraints, input data are not integrated to this repository. However, input and intermediary files required to run the analysis can be downloaded from a [figshare](https://doi.org/10.6084/m9.figshare.6934970). To run the following code, input and/or the intermediary files must be downloaded and placed in the folder. 
Then, do:

``` sh
createdb dsaa
gunzip < intermediate_db_backup.sql.gz | psql dsaa
tar -xf data.tar -C data/
```

Then place the content of dsaa_census_areas.zip into `data/generated_files/`.

To produce the intermediary files, go to the section "DIY Instructions".

# Code

The code of the analysis in divided in two parts: the Python scripts and modules used to support the analysis, and the notebooks where the outputs of the analysis have been produced.

## Scripts

* `data_processing_houses.ipynb` : script used for the pre-processing of Immobiliare.it data.
* `compute_walkability.py` : script used to generate the walkability scores for each census area.
* `data_processing_neighborhood.py` : script used to create all the dataset.
* `predict.py` : script used to predict the housing value from the intermediary files.
* `plots.ipynb` : script used to produce the images of the manuscript.


## License
This code is licensed under the MIT license. 


# DIY Instructions

Here we generate the entire database from ground. To do so, we have to create the minimal setup from this command:

``` sh
psql dsaa < data/SQL/minimal.sql
psql dsaa < data/SQL/minimal_materialize.sql
```

## Additional dependencies
* [osm2pgsql 0.95.0-dev](https://github.com/openstreetmap/osm2pgsql)
* [osrm v5.17.0](http://project-osrm.org/)
* [Security perception dependencies](https://github.com/denadai2/google_street_view_deep_neural)

## Census data
Census data have to complay to the format of the `census_areas_onfocus` table. Only when you did import data to this table you can proceed with all the steps. When you imported the data, you can generate the spatial matrix here:

``` sh
psql dsaa < data/SQL/first-DIY-step.sql
```


## Walkability
A OpenStreetMap file has to be downloaded (preferably from [here](https://wiki.openstreetmap.org/wiki/Planet.osm)), and placed in `data/OSM`. Then they are imported in PostGIS with:

``` sh
osm2pgsql -c -d dsaa --create --style "config/osm2pgsql.style" --multi-geometry --number-processes 5 --latlong -C 30000 [FILENAME].osm.pbf
```

The same file OSM file can then be used to produce the OSRM database:

``` sh
osrm-extract -p config/profiles/foot.lua [FILENAME].osm.pbf
osrm-contract [FILENAME].osrm
```

To run the server, use the command

``` sh
osrm-routed [FILENAME].osrm
```

After this everything is set up to create the intermediate data in the database. Import all the materialized view, then run the script. Before running it, personalize line 13 and 35 of `compute_walkability.py`.

``` sh
psql dsaa < data/SQL/walkability.sql
python3 compute_walkability.py
```

## Security perception
To create the security perception scores, we use the code and weights of the following paper:

* De Nadai, M., Vieriu, R. L., Zen, G., Dragicevic, S., Naik, N., Caraviello, M., ... & Lepri, B. *Are safer looking neighborhoods more lively?: A multimodal investigation into urban life*. In ACM MM 2016.

Everything is available [here](https://github.com/denadai2/google_street_view_deep_neural). All the prediction should be placed inside the `placepulse` table in PostgreSQL. Then, you can impor/refresh the materialized view present here:

``` sh
psql dsaa < data/SQL/security.sql
```

## Companies
You can insert a dataset with the census areas (`geoid`) and a proxy of companies earnings (`fatturato`) in `data/companies.csv`. Pay attention that this is included only in the non-open model version.

## Land value
You can insert a dataset with the census areas (`geoid`) and a proxy of land value (`assessed_land_value`) in `data/land_value.csv`. Pay attention that this is included only in the non-open model version.

## Census
Census data has to be inserted with the same format as the files placed in `data/census` and `data/census/industry`. To change this, change the corrisponding code at `data_processing_housing.py`.

## Land use
Download satellite shapefiles from https://land.copernicus.eu/local/urban-atlas/urban-atlas-2012/view. Import them in the `urban_atlas` PostgreSQL table. Then run the code:

``` sh
psql dsaa < data/SQL/urban_atlas.sql
```

# Some additional notes to the repository
* XGBoost 0.72 for some reason is not available anymore. I changed it to 0.71 because many users have contacted me because of this issue.