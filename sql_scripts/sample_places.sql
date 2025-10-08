-- SQL Script to Initialize Sample Data for Vietnam Travel MVP Chatbot and Test Queries
-- Sets 'id' as a BIGINT hash of the 'name' using MD5
-- Prevents duplicate entries in the 'places' table by checking for existing names
-- Assumes the 'places' table exists in the 'customer360' database with PostGIS extension enabled
-- Uses PostGIS functions: ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
-- Includes test queries to find the nearest place to given coordinates

-- Insert sample data, skipping records where the name already exists
INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Halong Bay'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Halong Bay', 
    'Halong City, Quang Ninh Province, Vietnam',
    'A UNESCO World Heritage site featuring thousands of limestone karsts and isles in various shapes and sizes, perfect for cruises and kayaking.', 
    'natural wonder', 
    ARRAY['bay', 'cruise', 'UNESCO', 'adventure'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(107.030322, 20.960429), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Halong Bay'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Hoi An Ancient Town'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Hoi An Ancient Town', 
    'Hoi An, Quang Nam Province, Vietnam',
    'A well-preserved ancient trading port with colorful lanterns, historic buildings, and a vibrant night market.', 
    'historical site', 
    ARRAY['ancient town', 'lanterns', 'UNESCO', 'culture'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(108.335, 15.87944), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Hoi An Ancient Town'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Phu Quoc Island'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Phu Quoc Island', 
    'Phu Quoc, Kien Giang Province, Vietnam',
    'Vietnam''s largest island known for pristine beaches, coral reefs, and national parks, ideal for relaxation and water activities.', 
    'island', 
    ARRAY['beach', 'resort', 'seafood', 'diving'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(103.95, 10.233), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Phu Quoc Island'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Sapa'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Sapa', 
    'Sapa Town, Lao Cai Province, Vietnam',
    'A mountainous region famous for terraced rice fields, ethnic minority villages, and trekking opportunities.', 
    'mountain town', 
    ARRAY['trekking', 'rice terraces', 'ethnic minorities', 'nature'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(103.873802, 22.356464), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Sapa'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Hanoi Old Quarter'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Hanoi Old Quarter', 
    'Hoan Kiem District, Hanoi, Vietnam',
    'The bustling heart of Hanoi with narrow streets, colonial architecture, street food, and traditional shops.', 
    'urban area', 
    ARRAY['old quarter', 'street food', 'historical', 'shopping'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(105.804817, 21.028511), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Hanoi Old Quarter'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Mekong Delta'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Mekong Delta', 
    'Can Tho City, Vietnam',
    'A vast network of rivers, floating markets, and rural villages showcasing Vietnam''s agricultural life.', 
    'river delta', 
    ARRAY['floating markets', 'boat tours', 'rural life', 'ecotourism'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(105.80, 10.04), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Mekong Delta'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Da Nang'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Da Nang', 
    'Da Nang City, Vietnam',
    'A modern coastal city with beautiful beaches, the famous Golden Bridge, and proximity to ancient sites.', 
    'city', 
    ARRAY['beach', 'bridges', 'modern', 'nightlife'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(108.206230, 16.047079), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Da Nang'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Hue Imperial City'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Hue Imperial City', 
    'Hue City, Thua Thien Hue Province, Vietnam',
    'The former imperial capital with a massive citadel, royal tombs, and pagodas reflecting Vietnam''s dynastic history.', 
    'historical site', 
    ARRAY['citadel', 'emperors', 'UNESCO', 'history'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(107.5833, 16.4667), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Hue Imperial City'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Nha Trang'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Nha Trang', 
    'Nha Trang City, Khanh Hoa Province, Vietnam',
    'A popular beach resort town with turquoise waters, islands, and opportunities for snorkeling and scuba diving.', 
    'coastal city', 
    ARRAY['beach', 'diving', 'islands', 'relaxation'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(109.1943, 12.2451), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Nha Trang'
);

INSERT INTO places (id, name, address, description, category, tags, pluscode, geom)
SELECT 
    ABS(CAST(CAST('x' || SUBSTR(MD5('Phong Nha-Ke Bang National Park'), 1, 16) AS BIT(64)) AS BIGINT)),
    'Phong Nha-Ke Bang National Park', 
    'Phong Nha, Quang Binh Province, Vietnam',
    'A UNESCO site renowned for its extensive cave systems, underground rivers, and karst landscapes.', 
    'national park', 
    ARRAY['caves', 'jungle', 'adventure', 'UNESCO'], 
    NULL, 
    ST_SetSRID(ST_MakePoint(106.112304, 17.5552645), 4326)
WHERE NOT EXISTS (
    SELECT 1 FROM places WHERE name = 'Phong Nha-Ke Bang National Park'
);

-- Insert sample system user, skipping if user_login already exists
INSERT INTO system_users (activation_key, avatar_url, creation_time, custom_data, display_name, is_online, modification_time, tenant_id, registered_time, role, status, user_email, user_login, user_pass, access_profile_fields, action_logs, in_groups, business_unit)
SELECT 
    'sample_activation_key_123', 
    'https://example.com/avatar.jpg', 
    1724600000000, 
    '{"department": "travel", "position": "guide"}', 
    'Travel Bot Admin', 
    FALSE, 
    1724600000000, 
    'vietnam_travel', 
    1724600000000, 
    4, 
    1, 
    'admin@travelbot.vn', 
    'admin', 
    'hashed_password_here', 
    ARRAY['profile', 'settings'], 
    ARRAY['logged_in'], 
    ARRAY['admins'], 
    'travel_ops'
WHERE NOT EXISTS (
    SELECT 1 FROM system_users WHERE user_login = 'admin'
);

-- Test Queries for Nearest Place
-- Test Query 1: Near Hanoi Old Quarter (approx lat: 21.028, long: 105.804)
SELECT id, 
       name, 
       address,
       description, 
       category, 
       tags,
       ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(105.804, 21.028), 4326)::geography) AS distance_meters
FROM places
ORDER BY distance_meters ASC
LIMIT 1;

-- Test Query 2: Near Da Nang (approx lat: 16.047, long: 108.206)
SELECT id, 
       name, 
       address,
       description, 
       category, 
       tags,
       ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(108.206, 16.047), 4326)::geography) AS distance_meters
FROM places
ORDER BY distance_meters ASC
LIMIT 1;

-- Test Query 3: Near Phu Quoc Island (approx lat: 10.233, long: 103.95)
SELECT id, 
       name, 
       address,
       description, 
       category, 
       tags,
       ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(103.95, 10.233), 4326)::geography) AS distance_meters
FROM places
ORDER BY distance_meters ASC
LIMIT 1;

-- Generic Query Template for Chatbot
/*
SELECT id, 
       name, 
       address,
       description, 
       category, 
       tags,
       ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(:user_long, :user_lat), 4326)::geography) AS distance_meters
FROM places
ORDER BY distance_meters ASC
LIMIT 1;
*/