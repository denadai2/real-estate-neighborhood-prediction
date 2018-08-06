create materialized view spatial_matrix as
select a.geoid as geoid1, b.geoid as geoid2
from census_areas_onfocus a
inner join census_areas_onfocus b ON st_dwithin(a.geom, b.geom, 0.01)
where a.geoid < b.geoid;
