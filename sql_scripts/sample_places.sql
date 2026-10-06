-- Sample places for the LEO BOT location-aware chatbot.
-- Coordinates are approximate WGS84 points (longitude, latitude).
-- Safe to rerun: place names are unique and are updated on conflict.

INSERT INTO places (
    name, address, description, category, tags, pluscode, geom
)
VALUES
(
    'Ben Thanh Market',
    'Le Loi, Ben Thanh Ward, District 1, Ho Chi Minh City',
    'Historic central market known for local food, souvenirs, clothing, and everyday goods.',
    'market',
    ARRAY['shopping', 'food', 'souvenirs', 'landmark'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6983, 10.7721), 4326)
),
(
    'Independence Palace',
    '135 Nam Ky Khoi Nghia, District 1, Ho Chi Minh City',
    'Landmark historic building and former presidential palace with preserved state rooms.',
    'historical site',
    ARRAY['history', 'architecture', 'museum', 'landmark'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6953, 10.7770), 4326)
),
(
    'War Remnants Museum',
    '28 Vo Van Tan, District 3, Ho Chi Minh City',
    'Museum documenting the history and consequences of the Vietnam War.',
    'museum',
    ARRAY['history', 'museum', 'culture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6921, 10.7794), 4326)
),
(
    'Saigon Central Post Office',
    '2 Cong Xa Paris, District 1, Ho Chi Minh City',
    'French-colonial landmark still operating as a post office beside Notre Dame Cathedral.',
    'historical site',
    ARRAY['architecture', 'photography', 'landmark'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6990, 10.7798), 4326)
),
(
    'Notre Dame Cathedral Basilica of Saigon',
    'Cong Xa Paris, District 1, Ho Chi Minh City',
    'Iconic red-brick cathedral in the city center.',
    'religious site',
    ARRAY['architecture', 'landmark', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6992, 10.7798), 4326)
),
(
    'Nguyen Hue Walking Street',
    'Nguyen Hue Boulevard, District 1, Ho Chi Minh City',
    'Pedestrian boulevard connecting City Hall with the Saigon River.',
    'public space',
    ARRAY['walking', 'nightlife', 'events', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7037, 10.7741), 4326)
),
(
    'Ho Chi Minh City Hall',
    '86 Le Thanh Ton, District 1, Ho Chi Minh City',
    'French-colonial civic building facing Nguyen Hue Walking Street.',
    'architecture',
    ARRAY['architecture', 'landmark', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7009, 10.7769), 4326)
),
(
    'Saigon Opera House',
    '7 Cong Truong Lam Son, District 1, Ho Chi Minh City',
    'Historic theater hosting concerts, ballet, opera, and cultural performances.',
    'theater',
    ARRAY['performance', 'culture', 'architecture', 'nightlife'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7031, 10.7769), 4326)
),
(
    'Bitexco Financial Tower',
    '2 Hai Trieu, District 1, Ho Chi Minh City',
    'Skyscraper with observation facilities and panoramic city views.',
    'observation deck',
    ARRAY['skyline', 'shopping', 'photography', 'landmark'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7042, 10.7717), 4326)
),
(
    'Landmark 81',
    '208 Nguyen Huu Canh, Binh Thanh District, Ho Chi Minh City',
    'Vietnam landmark skyscraper with dining, shopping, hotel, and observation views.',
    'observation deck',
    ARRAY['skyline', 'shopping', 'dining', 'landmark'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7216, 10.7950), 4326)
),
(
    'Bui Vien Walking Street',
    'Bui Vien, Pham Ngu Lao Ward, District 1, Ho Chi Minh City',
    'Busy nightlife street with restaurants, cafes, bars, and budget accommodation.',
    'nightlife',
    ARRAY['nightlife', 'food', 'bars', 'backpacking'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6927, 10.7687), 4326)
),
(
    'Tao Dan Park',
    'Truong Dinh, District 1, Ho Chi Minh City',
    'Large central green space with walking paths, gardens, and shaded rest areas.',
    'park',
    ARRAY['park', 'walking', 'nature', 'family'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6897, 10.7769), 4326)
),
(
    'Saigon Zoo and Botanical Gardens',
    '2 Nguyen Binh Khiem, District 1, Ho Chi Minh City',
    'Historic urban zoo and botanical garden near the city center.',
    'zoo and park',
    ARRAY['family', 'nature', 'animals', 'park'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7055, 10.7870), 4326)
),
(
    'Jade Emperor Pagoda',
    '73 Mai Thi Luu, Da Kao Ward, District 1, Ho Chi Minh City',
    'Famous Taoist pagoda with ornate statues, incense, and traditional architecture.',
    'religious site',
    ARRAY['pagoda', 'culture', 'architecture', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6961, 10.7904), 4326)
),
(
    'Tan Dinh Church',
    '289 Hai Ba Trung, District 3, Ho Chi Minh City',
    'Distinctive pink church known for its colorful facade and Gothic details.',
    'religious site',
    ARRAY['architecture', 'photography', 'landmark'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6860, 10.7875), 4326)
),
(
    'Vinh Nghiem Pagoda',
    '339 Nam Ky Khoi Nghia, District 3, Ho Chi Minh City',
    'Large Buddhist pagoda complex with a prominent seven-story tower.',
    'religious site',
    ARRAY['pagoda', 'culture', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6804, 10.7898), 4326)
),
(
    'Giac Lam Pagoda',
    '118 Lac Long Quan, Tan Binh District, Ho Chi Minh City',
    'One of the city''s oldest Buddhist temples with a peaceful garden setting.',
    'religious site',
    ARRAY['pagoda', 'history', 'culture', 'quiet'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6326, 10.7723), 4326)
),
(
    'Ba Thien Hau Temple',
    '710 Nguyen Trai, District 5, Ho Chi Minh City',
    'Historic Cantonese temple in Cho Lon dedicated to the sea goddess Thien Hau.',
    'religious site',
    ARRAY['temple', 'culture', 'history', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6601, 10.7531), 4326)
),
(
    'Binh Tay Market',
    '57A Thap Muoi, District 6, Ho Chi Minh City',
    'Large traditional wholesale market and a major landmark of Cho Lon.',
    'market',
    ARRAY['shopping', 'food', 'culture', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6513, 10.7528), 4326)
),
(
    'Phuoc An Hoi Quan Assembly Hall',
    '184 Hong Bang, District 5, Ho Chi Minh City',
    'Ornate Chinese assembly hall with colorful woodwork, statues, and lanterns.',
    'cultural site',
    ARRAY['culture', 'architecture', 'temple', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6595, 10.7523), 4326)
),
(
    'Cho Lon Chinatown',
    'Cho Lon, Districts 5 and 6, Ho Chi Minh City',
    'Historic Chinese-Vietnamese neighborhood known for markets, temples, and food.',
    'neighborhood',
    ARRAY['culture', 'food', 'markets', 'walking'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6570, 10.7540), 4326)
),
(
    'Ong Bon Pagoda',
    '264 Hai Thuong Lan Ong, District 5, Ho Chi Minh City',
    'Historic Chinese temple known for traditional decoration and community festivals.',
    'religious site',
    ARRAY['temple', 'culture', 'history', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6576, 10.7537), 4326)
),
(
    'Mariamman Hindu Temple',
    '45 Truong Dinh, District 1, Ho Chi Minh City',
    'Colorful Hindu temple in the central city with a detailed entrance tower.',
    'religious site',
    ARRAY['temple', 'culture', 'architecture', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6963, 10.7718), 4326)
),
(
    'Ho Chi Minh City Fine Arts Museum',
    '97A Pho Duc Chinh, District 1, Ho Chi Minh City',
    'Museum in a historic mansion featuring Vietnamese modern and traditional art.',
    'museum',
    ARRAY['art', 'museum', 'architecture', 'culture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6981, 10.7687), 4326)
),
(
    'Ho Chi Minh Museum at Nha Rong Wharf',
    '1 Nguyen Tat Thanh, District 4, Ho Chi Minh City',
    'Museum and historic riverside landmark associated with Ho Chi Minh''s journey.',
    'museum',
    ARRAY['history', 'museum', 'river', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7060, 10.7686), 4326)
),
(
    'Ho Chi Minh City Museum',
    '65 Ly Tu Trong, District 1, Ho Chi Minh City',
    'Historic neoclassical building with exhibitions about the city and region.',
    'museum',
    ARRAY['history', 'museum', 'architecture', 'culture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6985, 10.7768), 4326)
),
(
    'Nguyen Van Binh Book Street',
    'Nguyen Van Binh, District 1, Ho Chi Minh City',
    'Pedestrian book street with bookstores, reading spaces, and cafes.',
    'public space',
    ARRAY['books', 'walking', 'cafes', 'family'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6995, 10.7794), 4326)
),
(
    'The Cafe Apartments',
    '42 Nguyen Hue, District 1, Ho Chi Minh City',
    'Repurposed apartment building filled with cafes, boutiques, and creative spaces.',
    'shopping and dining',
    ARRAY['cafes', 'shopping', 'photography', 'walking'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7036, 10.7741), 4326)
),
(
    'Golden Dragon Water Puppet Theater',
    '55B Nguyen Thi Minh Khai, District 1, Ho Chi Minh City',
    'Traditional Vietnamese water puppet theater with regular cultural performances.',
    'theater',
    ARRAY['culture', 'family', 'performance', 'traditional art'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6930, 10.7778), 4326)
),
(
    'Saigon River Cruise Pier',
    'Ton Duc Thang, District 1, Ho Chi Minh City',
    'Riverside departure area for sightseeing cruises and evening meals on the river.',
    'river attraction',
    ARRAY['river', 'cruise', 'nightlife', 'views'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7048, 10.7710), 4326)
),
(
    'Vinhomes Central Park',
    '208 Nguyen Huu Canh, Binh Thanh District, Ho Chi Minh City',
    'Large riverside urban park with lawns, gardens, walking paths, and skyline views.',
    'park',
    ARRAY['park', 'walking', 'family', 'river'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7210, 10.7945), 4326)
),
(
    'Binh Quoi Tourist Village',
    '1147 Binh Quoi, Binh Thanh District, Ho Chi Minh City',
    'Riverside recreational area with gardens, local food, and traditional village scenery.',
    'leisure area',
    ARRAY['river', 'family', 'food', 'nature'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7380, 10.8170), 4326)
),
(
    'Crescent Mall',
    '101 Ton Dat Tien, District 7, Ho Chi Minh City',
    'Modern shopping mall with retail, restaurants, cinema, and family entertainment.',
    'shopping mall',
    ARRAY['shopping', 'dining', 'cinema', 'family'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7183, 10.7297), 4326)
),
(
    'Starlight Bridge',
    'Ton Dat Tien, District 7, Ho Chi Minh City',
    'Illuminated pedestrian bridge near Crescent Lake and the Phu My Hung urban area.',
    'public space',
    ARRAY['walking', 'photography', 'nightlife', 'family'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7179, 10.7298), 4326)
),
(
    'Saigon Exhibition and Convention Center',
    '799 Nguyen Van Linh, District 7, Ho Chi Minh City',
    'Major venue for exhibitions, conferences, trade fairs, and public events.',
    'event venue',
    ARRAY['events', 'exhibitions', 'business'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7210, 10.7290), 4326)
),
(
    'SC VivoCity',
    '1058 Nguyen Van Linh, District 7, Ho Chi Minh City',
    'Shopping and entertainment complex with restaurants, cinema, and family activities.',
    'shopping mall',
    ARRAY['shopping', 'dining', 'cinema', 'family'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7046, 10.7314), 4326)
),
(
    'Dam Sen Cultural Park',
    '3 Hoa Binh, District 11, Ho Chi Minh City',
    'Large amusement and cultural park with gardens, rides, lakes, and family attractions.',
    'theme park',
    ARRAY['family', 'rides', 'water park', 'nature'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6350, 10.7663), 4326)
),
(
    'Suoi Tien Theme Park',
    '120 Hanoi Highway, Thu Duc City, Ho Chi Minh City',
    'Large cultural amusement park with rides, water attractions, and Vietnamese themes.',
    'theme park',
    ARRAY['family', 'rides', 'water park', 'entertainment'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.8027, 10.8664), 4326)
),
(
    'Tan Son Nhat International Airport',
    'Truong Son, Tan Binh District, Ho Chi Minh City',
    'The city''s main airport and a useful reference point for travelers.',
    'transport hub',
    ARRAY['airport', 'travel', 'transport'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6519, 10.8188), 4326)
),
(
    'Cu Chi Tunnels',
    'Phu Hiep, Cu Chi District, Ho Chi Minh City',
    'Historic underground tunnel network and wartime heritage attraction outside the center.',
    'historical site',
    ARRAY['history', 'tunnels', 'adventure', 'museum'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.4630, 11.1410), 4326)
),
(
    'Can Gio Mangrove Biosphere Reserve',
    'Can Gio District, Ho Chi Minh City',
    'Coastal mangrove reserve with wetlands, wildlife, and eco-tourism activities.',
    'nature reserve',
    ARRAY['nature', 'mangrove', 'wildlife', 'eco-tourism'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.9540, 10.4110), 4326)
),
(
    'Thu Thiem Urban Area',
    'Thu Thiem, Thu Duc City, Ho Chi Minh City',
    'Modern riverside urban district with open spaces and views toward District 1.',
    'urban area',
    ARRAY['river', 'skyline', 'walking', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7215, 10.7750), 4326)
),
(
    'Thao Dien',
    'Thao Dien Ward, Thu Duc City, Ho Chi Minh City',
    'Popular riverside neighborhood with restaurants, cafes, galleries, and boutiques.',
    'neighborhood',
    ARRAY['cafes', 'dining', 'shopping', 'nightlife'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7330, 10.8010), 4326)
),
(
    'Le Van Tam Park',
    'Vo Thi Sau, District 1, Ho Chi Minh City',
    'Central green park with walking paths, open lawns, and community activities.',
    'park',
    ARRAY['park', 'walking', 'family', 'nature'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6880, 10.7880), 4326)
),
(
    'Hoang Van Thu Park',
    'Hoang Van Thu, Tan Binh District, Ho Chi Minh City',
    'Large landscaped park near the airport with open lawns and walking paths.',
    'park',
    ARRAY['park', 'walking', 'family', 'nature'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6620, 10.8060), 4326)
),
(
    'Saigon Central Mosque',
    '66 Dong Du, District 1, Ho Chi Minh City',
    'Historic mosque and quiet courtyard near the central shopping and hotel area.',
    'religious site',
    ARRAY['mosque', 'architecture', 'culture', 'quiet'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7040, 10.7740), 4326)
),
(
    'Saigon Skydeck',
    'Bitexco Financial Tower, 2 Hai Trieu, District 1, Ho Chi Minh City',
    'Observation deck offering panoramic views of central Ho Chi Minh City.',
    'observation deck',
    ARRAY['skyline', 'views', 'photography', 'landmark'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7043, 10.7718), 4326)
),
(
    'Tan Cang Riverside',
    '100 Ung Van Khiem, Binh Thanh District, Ho Chi Minh City',
    'Riverside leisure area with dining, open views, and access to the Saigon waterfront.',
    'riverside area',
    ARRAY['river', 'dining', 'views', 'relaxation'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7160, 10.7930), 4326)
),
(
    'Viet Nam Quoc Tu Pagoda',
    '244 Ba Thang Hai, District 10, Ho Chi Minh City',
    'Large Buddhist pagoda with a tall tower and a prominent urban location.',
    'religious site',
    ARRAY['pagoda', 'culture', 'architecture', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6720, 10.7720), 4326)
),
(
    'Xa Loi Pagoda',
    '89 Ba Huyen Thanh Quan, District 3, Ho Chi Minh City',
    'Historic Buddhist pagoda with a bell tower and peaceful courtyard.',
    'religious site',
    ARRAY['pagoda', 'history', 'culture', 'quiet'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6870, 10.7790), 4326)
)
ON CONFLICT (name) DO UPDATE SET
    address = EXCLUDED.address,
    description = EXCLUDED.description,
    category = EXCLUDED.category,
    tags = EXCLUDED.tags,
    pluscode = EXCLUDED.pluscode,
    geom = EXCLUDED.geom,
    updated_at = NOW();
