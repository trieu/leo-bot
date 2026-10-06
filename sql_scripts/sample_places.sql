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
),
(
    'Huyen Sy Church',
    '1 Ton That Tung, District 1, Ho Chi Minh City',
    'Historic Catholic church known for its Gothic architecture and central location.',
    'church',
    ARRAY['church', 'Catholic', 'architecture', 'history'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6857, 10.7653), 4326)
),
(
    'Cho Quan Church',
    '120 Tran Binh Trong, District 5, Ho Chi Minh City',
    'Historic Catholic parish church serving the Cho Quan area.',
    'church',
    ARRAY['church', 'Catholic', 'architecture', 'history'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6769, 10.7548), 4326)
),
(
    'Cha Tam Church',
    '25 Hoc Lac, District 5, Ho Chi Minh City',
    'Historic Chinese Catholic church in the Cho Lon neighborhood.',
    'church',
    ARRAY['church', 'Catholic', 'Cho Lon', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6588, 10.7517), 4326)
),
(
    'St Jeanne d''Arc Church',
    '116B Hung Vuong, District 5, Ho Chi Minh City',
    'Distinctive Catholic church commonly known as the Nga Sau Church.',
    'church',
    ARRAY['church', 'Catholic', 'architecture', 'photography'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6820, 10.7680), 4326)
),
(
    'Hanh Thong Tay Church',
    '7 Quang Trung, Go Vap District, Ho Chi Minh City',
    'Popular Catholic church serving the Hanh Thong Tay neighborhood.',
    'church',
    ARRAY['church', 'Catholic', 'community', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6650, 10.8420), 4326)
),
(
    'Go Vap Church',
    '535 Quang Trung, Go Vap District, Ho Chi Minh City',
    'Local Catholic church and community landmark in Go Vap.',
    'church',
    ARRAY['church', 'Catholic', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6660, 10.8370), 4326)
),
(
    'Phu Nhuan Church',
    '386 Nguyen Kiem, Phu Nhuan District, Ho Chi Minh City',
    'Catholic parish church near the Phu Nhuan urban center.',
    'church',
    ARRAY['church', 'Catholic', 'community', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6790, 10.7990), 4326)
),
(
    'Vuon Xoai Church',
    '161B Le Van Sy, District 3, Ho Chi Minh City',
    'Catholic church known for parish activities and a welcoming urban campus.',
    'church',
    ARRAY['church', 'Catholic', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6800, 10.7860), 4326)
),
(
    'Ky Dong Church',
    '40 Ky Dong, District 3, Ho Chi Minh City',
    'Catholic church and pilgrimage destination in central District 3.',
    'church',
    ARRAY['church', 'Catholic', 'pilgrimage', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6850, 10.7850), 4326)
),
(
    'Tan Huong Church',
    '117 Tan Huong, Tan Phu District, Ho Chi Minh City',
    'Catholic parish church serving the Tan Huong neighborhood.',
    'church',
    ARRAY['church', 'Catholic', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6270, 10.7890), 4326)
),
(
    'Tan Phu Church',
    '158 Nguyen Son, Tan Phu District, Ho Chi Minh City',
    'Catholic church serving families and parish communities in Tan Phu.',
    'church',
    ARRAY['church', 'Catholic', 'community', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6250, 10.7900), 4326)
),
(
    'Phu Tho Church',
    '18 Nguyen Thi Nho, District 10, Ho Chi Minh City',
    'Catholic parish church near the Phu Tho area and local markets.',
    'church',
    ARRAY['church', 'Catholic', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6600, 10.7720), 4326)
),
(
    'Binh Thai Church',
    '173A Pham Phu Thu, District 6, Ho Chi Minh City',
    'Catholic church serving the Binh Thai community near Cho Lon.',
    'church',
    ARRAY['church', 'Catholic', 'Cho Lon', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6570, 10.7560), 4326)
),
(
    'Binh An Church',
    '178 Nguyen Thi Thap, District 7, Ho Chi Minh City',
    'Catholic parish church in the growing District 7 urban area.',
    'church',
    ARRAY['church', 'Catholic', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7240, 10.7410), 4326)
),
(
    'Thanh Da Church',
    'Thanh Da Peninsula, Binh Thanh District, Ho Chi Minh City',
    'Riverside Catholic church serving the Thanh Da neighborhood.',
    'church',
    ARRAY['church', 'Catholic', 'river', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7280, 10.8080), 4326)
),
(
    'Thu Duc Church',
    '22 Vo Van Ngan, Thu Duc City, Ho Chi Minh City',
    'Historic Catholic parish church in the center of Thu Duc.',
    'church',
    ARRAY['church', 'Catholic', 'architecture', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7560, 10.8490), 4326)
),
(
    'Thu Thiem Church',
    'Thu Thiem Ward, Thu Duc City, Ho Chi Minh City',
    'Historic riverside Catholic church in the Thu Thiem area.',
    'church',
    ARRAY['church', 'Catholic', 'history', 'river'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7300, 10.7740), 4326)
),
(
    'Fatima Binh Trieu Church',
    'Binh Trieu, Thu Duc City, Ho Chi Minh City',
    'Popular Catholic pilgrimage and parish site near the Saigon River.',
    'church',
    ARRAY['church', 'Catholic', 'pilgrimage', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7350, 10.8530), 4326)
),
(
    'Binh Loi Church',
    'Binh Loi, Binh Thanh District, Ho Chi Minh City',
    'Catholic church serving the riverside Binh Loi community.',
    'church',
    ARRAY['church', 'Catholic', 'river', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7200, 10.8200), 4326)
),
(
    'Ba Diem Church',
    'Ba Diem, Hoc Mon District, Ho Chi Minh City',
    'Catholic parish church serving the Ba Diem community northwest of the city.',
    'church',
    ARRAY['church', 'Catholic', 'community', 'history'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.5950, 10.8670), 4326)
),
(
    'Bien Hoa Cathedral',
    '57 Nguyen Ai Quoc, Bien Hoa City, Dong Nai Province',
    'Major Catholic cathedral serving Bien Hoa and surrounding communities.',
    'church',
    ARRAY['church', 'Catholic', 'cathedral', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.8430, 10.9570), 4326)
),
(
    'Tan Mai Church',
    'Tan Mai Ward, Bien Hoa City, Dong Nai Province',
    'Large Catholic parish church in the Tan Mai area of Bien Hoa.',
    'church',
    ARRAY['church', 'Catholic', 'community', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.8500, 10.9400), 4326)
),
(
    'Tam Hiep Church',
    'Tam Hiep Ward, Bien Hoa City, Dong Nai Province',
    'Catholic church serving a large suburban parish east of Ho Chi Minh City.',
    'church',
    ARRAY['church', 'Catholic', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.8550, 10.9400), 4326)
),
(
    'Phu Cuong Cathedral',
    '6 Nguyen Truong To, Thu Dau Mot City, Binh Duong Province',
    'Prominent Catholic cathedral and architectural landmark in Thu Dau Mot.',
    'church',
    ARRAY['church', 'Catholic', 'cathedral', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.6540, 10.9800), 4326)
),
(
    'Lai Thieu Church',
    'Lai Thieu Ward, Thuan An City, Binh Duong Province',
    'Historic Catholic church serving the Lai Thieu community north of the city.',
    'church',
    ARRAY['church', 'Catholic', 'history', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(106.7040, 10.9030), 4326)
),
(
    'Ba Ria Cathedral',
    'Nguyen Tat Thanh, Ba Ria City, Ba Ria - Vung Tau Province',
    'Catholic cathedral serving Ba Ria and the surrounding coastal province.',
    'church',
    ARRAY['church', 'Catholic', 'cathedral', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(107.1680, 10.4950), 4326)
),
(
    'Vung Tau Cathedral',
    'Tran Hung Dao, Vung Tau City, Ba Ria - Vung Tau Province',
    'Central Catholic cathedral in Vung Tau near the city waterfront.',
    'church',
    ARRAY['church', 'Catholic', 'cathedral', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(107.0840, 10.3460), 4326)
),
(
    'Long Hai Church',
    'Long Hai, Long Dien District, Ba Ria - Vung Tau Province',
    'Catholic parish church near the Long Hai coastal area.',
    'church',
    ARRAY['church', 'Catholic', 'coast', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(107.2400, 10.3860), 4326)
),
(
    'Phuoc Hai Church',
    'Phuoc Hai, Dat Do District, Ba Ria - Vung Tau Province',
    'Catholic church serving the fishing and coastal community of Phuoc Hai.',
    'church',
    ARRAY['church', 'Catholic', 'coast', 'community'],
    NULL,
    ST_SetSRID(ST_MakePoint(107.2600, 10.4080), 4326)
),
(
    'Xuan Loc Cathedral',
    'Xuan Loc City, Dong Nai Province',
    'Catholic cathedral and diocesan landmark east of Ho Chi Minh City.',
    'church',
    ARRAY['church', 'Catholic', 'cathedral', 'architecture'],
    NULL,
    ST_SetSRID(ST_MakePoint(107.2440, 10.9290), 4326)
)
ON CONFLICT (name) DO UPDATE SET
    address = EXCLUDED.address,
    description = EXCLUDED.description,
    category = EXCLUDED.category,
    tags = EXCLUDED.tags,
    pluscode = EXCLUDED.pluscode,
    geom = EXCLUDED.geom,
    updated_at = NOW();
