CREATE MATERIALIZED VIEW urban_atlas_4326 AS
select code2012, ST_Transform(wkb_geometry, 4326) as geom
FROM urban_atlas;
CREATE INDEX ON urban_atlas_4326 (code2012);
CREATE INDEX ON urban_atlas_4326 USING GIST (geom);


CREATE MATERIALIZED VIEW urban_atlas_census_dataset AS
select geoid, COALESCE(SUM(CASE WHEN c.code2012 IN('11100', '11210', '11220', '11230', '11240', '11300') THEN area END), 0) as urban_sum,
    COALESCE(SUM(CASE WHEN c.code2012 IN('12100', '13100') THEN area END), 0) as comm_sum,
    COALESCE(SUM(CASE WHEN c.code2012 IN('14100', '14200') THEN area END), 0) as green_sum,
    COALESCE(SUM(CASE WHEN c.code2012 IN('12210', '12220', '12230') THEN area END), 0) as roads_sum,
    COALESCE(SUM(CASE WHEN c.code2012 IN('21000', '22000', '23000', '24000', '25000', '31000', '32000', '33000', '40000', '50000') THEN area END), 0) as other_sum
FROM urban_atlas_census c
GROUP BY geoid;


copy (select * from urban_atlas_census_dataset) TO 'data/generated_files/urban_atlas_census.csv' DELIMITER ',' CSV HEADER;