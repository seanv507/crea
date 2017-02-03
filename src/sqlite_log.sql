CREATE TABLE report_20160201_20161201_a ( 
	row_id INTEGER PRIMARY KEY, 
	ad_group_id INTEGER,
	ad_group INTEGER,
	campaign_id INTEGER,
	campaign INTEGER,
	countryterritory INTEGER,
	day REAL,
	device INTEGER,
	category_1st_level INTEGER,
	category_2nd_level INTEGER,
	category_3rd_level INTEGER,
	category_4th_level INTEGER,
	category_5th_level INTEGER,
	product_type_1st_level INTEGER,
	product_type_2nd_level INTEGER,
	product_type_3rd_level INTEGER,
	product_type_4th_level INTEGER,
	product_type_5th_level INTEGER,
	custom_label_0 INTEGER,
	custom_label_1 INTEGER,
	custom_label_2 INTEGER,
	custom_label_3 INTEGER,
	custom_label_4 INTEGER,
	brand INTEGER,
	item_id INTEGER,
	impressions INTEGER,
	clicks INTEGER,
	conversions INTEGER,
	cross_device_conv INTEGER,
	total_conv_value REAL,
	cost REAL,
	click_share TEXT,
	search_impr_share TEXT,
	ctr REAL,conv_rate REAL)
	
INSERT INTO 
	report_20160201_20161201_a
SELECT	
	row_id,
	ad_group_id,
	ad_group,
	campaign_id ,
	campaign,
	countryterritory,
	julianday(day),
	device,
	category_1st_level,
	category_2nd_level,
	category_3rd_level,
	category_4th_level,
	category_5th_level,
	product_type_1st_level,
	product_type_2nd_level,
	product_type_3rd_level,
	product_type_4th_level,
	product_type_5th_level,
	custom_label_0,
	custom_label_1,
	custom_label_2,
	custom_label_3,
	custom_label_4,
	brand,
	item_id,
	impressions,
	clicks,
	conversions,
	cross_device_conv,
	total_conv_value,
	cost,
	click_share,
	search_impr_share,
	ctr,
	conv_rate
FROM
	report_20160201_20161201
	
Query executed successfully: CREATE INDEX ix_day ON report_20160201_20161201(day) (took 321027ms)
select 
	category_1st_level,
	category_2nd_level,
	category_3rd_level,
	category_4th_level,
	category_5th_level,
	count(*)
from report_20160201_20161201
group by
	category_1st_level,
	category_2nd_level,
	category_3rd_level,
	category_4th_level,
	category_5th_level
select * from lookup_20160201_20161201
where id_name='Category (2nd level)' and id_index=5
select * from lookup_20160201_20161201
where id_name='Category (3nd level)' and id_index=9
select * from lookup_20160201_20161201
where id_name='Category (3rd level)' and id_index=9
select * from lookup_20160201_20161201
where id_name='Category (4th level)' and id_index=0
