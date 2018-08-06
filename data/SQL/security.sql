CREATE MATERIALIZED VIEW placepulse_geom AS
SELECT prediction, ST_Buffer(ST_setsrid(ST_Makepoint(lon, lat), 4326), 0.0005) as geom
FROM placepulse;
CREATE INDEX ON placepulse_geom using GIST (geom);

CREATE MATERIALIZED VIEW placepulse_census AS
SELECT geoid, AVG(prediction) as score
FROM census_areas_onfocus as c
INNER JOIN placepulse_geom p ON ST_Intersects(p.geom, c.geom)
GROUP BY geoid;


copy (select * from placepulse_census) TO 'data/generated_files/placepulse_census.csv' DELIMITER ',' CSV HEADER;

