--
-- PostgreSQL database dump
--

-- Dumped from database version 10.0
-- Dumped by pg_dump version 10.0

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


--
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


--
-- Name: EXTENSION postgis; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION postgis IS 'PostGIS geometry, geography, and raster spatial types and functions';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: census_areas_onfocus; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE census_areas_onfocus (
    geoid character varying(13),
    geom geometry(MultiPolygon,4326),
    clon double precision,
    clat double precision
);


--
-- Name: cities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE cities (
    pro_com integer NOT NULL,
    name character varying(255),
    note character varying(250)
);


--
-- Name: sea; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE sea (
    gid integer NOT NULL,
    fid double precision,
    fid_2 numeric,
    "@id" character varying(254),
    "natural" character varying(254),
    official_n character varying(254),
    type character varying(254),
    water character varying(254),
    wikipedia character varying(254),
    shallow character varying(254),
    "@id_2" character varying(254),
    natural_2 character varying(254),
    official_2 character varying(254),
    type_2 character varying(254),
    water_2 character varying(254),
    wikipedi_2 character varying(254),
    shallow_2 character varying(254),
    geom geometry(MultiPolygon,4326)
);


--
-- Name: sea_gid_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE sea_gid_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sea_gid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE sea_gid_seq OWNED BY sea.gid;


--
-- Name: urban_atlas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE urban_atlas (
    ogc_fid integer NOT NULL,
    country character varying(50),
    cities character varying(254),
    fua_or_cit character varying(254),
    code2012 character varying(7),
    item2012 character varying(150),
    prod_date character varying(4),
    ident character varying(30),
    shape_leng numeric(19,11),
    shape_area numeric(19,11),
    wkb_geometry geometry(Geometry,900914)
);


--
-- Name: urban_atlas_ogc_fid_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE urban_atlas_ogc_fid_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: urban_atlas_ogc_fid_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE urban_atlas_ogc_fid_seq OWNED BY urban_atlas.ogc_fid;


--
-- Name: sea gid; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY sea ALTER COLUMN gid SET DEFAULT nextval('sea_gid_seq'::regclass);


--
-- Name: urban_atlas ogc_fid; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY urban_atlas ALTER COLUMN ogc_fid SET DEFAULT nextval('urban_atlas_ogc_fid_seq'::regclass);


--
-- Name: cities cities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY cities
    ADD CONSTRAINT cities_pkey PRIMARY KEY (pro_com);


--
-- Name: sea sea_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY sea
    ADD CONSTRAINT sea_pkey PRIMARY KEY (gid);


--
-- Name: urban_atlas urban_atlas_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY urban_atlas
    ADD CONSTRAINT urban_atlas_pkey PRIMARY KEY (ogc_fid);


--
-- Name: census_areas_onfocus_clat_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX census_areas_onfocus_clat_idx ON census_areas_onfocus USING btree (clat);


--
-- Name: census_areas_onfocus_clon_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX census_areas_onfocus_clon_idx ON census_areas_onfocus USING btree (clon);


--
-- Name: census_areas_onfocus_geoid_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX census_areas_onfocus_geoid_idx ON census_areas_onfocus USING btree (geoid);


--
-- Name: census_areas_onfocus_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX census_areas_onfocus_geom_idx ON census_areas_onfocus USING gist (geom);


--
-- Name: cities_name_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX cities_name_idx ON cities USING btree (name);


--
-- Name: urban_atlas_code2012_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX urban_atlas_code2012_idx ON urban_atlas USING btree (code2012);


--
-- Name: urban_atlas_wkb_geometry_geom_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX urban_atlas_wkb_geometry_geom_idx ON urban_atlas USING gist (wkb_geometry);


--
-- PostgreSQL database dump complete
--

