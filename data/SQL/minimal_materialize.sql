
--
-- Name: planet_osm_line_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_line_index ON planet_osm_line USING gist (way) WITH (fillfactor='100');


--
-- Name: planet_osm_point_aeroway_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_aeroway_idx ON planet_osm_point USING hash (aeroway);


--
-- Name: planet_osm_point_amenity_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_amenity_idx ON planet_osm_point USING hash (amenity);


--
-- Name: planet_osm_point_geography_aeroway_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_geography_aeroway_idx ON planet_osm_point_geography USING hash (aeroway);


--
-- Name: planet_osm_point_geography_amenity_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_geography_amenity_idx ON planet_osm_point_geography USING hash (amenity);


--
-- Name: planet_osm_point_geography_gway_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_geography_gway_idx ON planet_osm_point_geography USING gist (gway);


--
-- Name: planet_osm_point_geography_railway_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_geography_railway_idx ON planet_osm_point_geography USING hash (railway);


--
-- Name: planet_osm_point_geography_shop_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_geography_shop_idx ON planet_osm_point_geography USING btree (shop);


--
-- Name: planet_osm_point_geography_way_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_geography_way_idx ON planet_osm_point_geography USING gist (way);


--
-- Name: planet_osm_point_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_index ON planet_osm_point USING gist (way) WITH (fillfactor='100');


--
-- Name: planet_osm_point_railway_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_railway_idx ON planet_osm_point USING hash (railway);


--
-- Name: planet_osm_point_shop_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_shop_idx ON planet_osm_point USING btree (shop);


--
-- Name: planet_osm_point_station_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_station_idx ON planet_osm_point USING hash (station);


--
-- Name: planet_osm_point_station_idx1; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_station_idx1 ON planet_osm_point USING btree (station) WHERE (station = 'subway'::text);


--
-- Name: planet_osm_point_tourism_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_point_tourism_idx ON planet_osm_point USING hash (tourism);


--
-- Name: planet_osm_polygon_amenity_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_polygon_amenity_idx ON planet_osm_polygon USING hash (amenity);


--
-- Name: planet_osm_polygon_building_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_polygon_building_idx ON planet_osm_polygon USING btree (building);


--
-- Name: planet_osm_polygon_building_landuse_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_polygon_building_landuse_idx ON planet_osm_polygon USING btree (building, landuse) WHERE ((landuse = 'industrial'::text) AND (building = 'industrial'::text));


--
-- Name: planet_osm_polygon_building_landuse_idx1; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_polygon_building_landuse_idx1 ON planet_osm_polygon USING btree (building, landuse);


--
-- Name: planet_osm_polygon_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_polygon_index ON planet_osm_polygon USING gist (way) WITH (fillfactor='100');


--
-- Name: planet_osm_polygon_landuse_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_polygon_landuse_idx ON planet_osm_polygon USING hash (landuse);


--
-- Name: planet_osm_polygon_leisure_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_polygon_leisure_idx ON planet_osm_polygon USING hash (leisure);


--
-- Name: planet_osm_roads_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX planet_osm_roads_index ON planet_osm_roads USING gist (way) WITH (fillfactor='100');



CREATE MATERIALIZED VIEW osm_walkability_nbus_stops AS
SELECT c.geoid, COUNT(p.osm_id) as c
FROM census_areas_onfocus c
LEFT JOIN planet_osm_point p ON ST_DWithin(c.geom, p.way, 0.005) AND (highway='bus_stop' OR public_transport='stop_position')
GROUP BY geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_airport AS
SELECT c.geoid, near_point.dist
FROM census_areas_onfocus c,
LATERAL (SELECT ROUND(ST_Distance(c.geom::geography, p.way::geography)) as dist
         FROM planet_osm_point p
         WHERE aeroway='terminal'
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point;


CREATE MATERIALIZED VIEW osm_walkability_prox_cache_industrial AS
SELECT p.way, p.way::geography as gway
FROM planet_osm_polygon p
WHERE landuse='industrial';
CREATE INDEX ON osm_walkability_prox_cache_industrial using gist(way);

CREATE MATERIALIZED VIEW osm_walkability_prox_industrial_area AS
SELECT c.geoid, near_point.dist
FROM census_areas_onfocus c,
LATERAL (SELECT ROUND(ST_Distance(c.geom::geography, p.gway)) as dist
         FROM osm_walkability_prox_cache_industrial p
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point;


CREATE MATERIALIZED VIEW osm_walkability_prox_grocery AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest
         FROM planet_osm_point p
         WHERE shop IN ('supermarket', 'convenience', 'grocery', 'greengrocer', 'bakery', 'butcher', 'beverages', 'cheese', 'deli', 'dairy', 'health_food', 'pasta', 'seafood', 'spices')
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_restaurants AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest
         FROM planet_osm_point p
         WHERE amenity='restaurant' OR amenity IN('fast_food', 'bar', 'pub')
         ORDER BY c.geom <-> p.way
         LIMIT 10) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_cache_shops AS
SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest, p.way
FROM planet_osm_point p
WHERE shop IS NOT NULL AND shop NOT IN ('supermarket', 'convenience', 'grocery', 'greengrocer', 'bakery', 'butcher', 'beverages', 'cheese', 'deli', 'dairy', 'health_food', 'pasta', 'seafood', 'spices');
CREATE INDEX ON osm_walkability_prox_cache_shops using gist(way);


CREATE MATERIALIZED VIEW osm_walkability_prox_shops AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT p.dest
         FROM osm_walkability_prox_cache_shops p
         ORDER BY c.geom <-> p.way
         LIMIT 5) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_school AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest
         FROM planet_osm_point p
         WHERE amenity IN ('school', 'university', 'college', 'kindergarten')
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_enter AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest
         FROM planet_osm_point p
         WHERE amenity IN('cinema', 'theatre', 'arts_centre')
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_coffee AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest
         FROM planet_osm_point p
         WHERE amenity='cafe'
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_cache_metro AS
SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest, p.way
FROM planet_osm_point p
WHERE station='subway';
CREATE INDEX ON osm_walkability_prox_cache_metro using gist(way);


CREATE MATERIALIZED VIEW osm_walkability_prox_metro AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest
         FROM osm_walkability_prox_cache_metro p
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_library AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest
         FROM planet_osm_point p
         WHERE amenity = 'library' OR tourism='museum'
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_railway AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(p.way)::text||','||ST_Y(p.way)::text as dest
         FROM planet_osm_point p
         WHERE railway='station' AND (station <> 'subway' OR station IS NULL)
         ORDER BY c.geom <-> p.way
         LIMIT 1) near_point
group by geoid;


CREATE MATERIALIZED VIEW osm_walkability_prox_parks AS
SELECT c.geoid, array_agg(dest) as dests
FROM census_areas_onfocus c,
LATERAL (SELECT ST_X(ST_Centroid(p.way))::text||','||ST_Y(ST_Centroid(p.way))::text as dest
         FROM planet_osm_polygon p
         WHERE amenity='park' OR landuse='park' OR leisure='park'
         ORDER BY c.geom <-> p.way
         LIMIT 2) near_point
group by geoid;



CREATE TABLE placepulse (dir text, filename text,lat float, lon float, label float,prediction float);


CREATE MATERIALIZED VIEW placepulse_geom AS
SELECT prediction, ST_Buffer(ST_setsrid(ST_Makepoint(lon, lat), 4326), 0.0005) as geom
FROM placepulse;
CREATE INDEX ON placepulse_geom using GIST (geom);


CREATE MATERIALIZED VIEW placepulse_census AS
SELECT geoid, AVG(prediction) as score
FROM census_areas as c
INNER JOIN placepulse_geom p ON ST_Intersects(p.geom, c.geom)
GROUP BY geoid;