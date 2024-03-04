from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, Float
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime

# Database connection
engine = create_engine('sqlite:///tours.db')

# Base class for all models
Base = declarative_base()


class City(Base):
    __tablename__ = 'cities'

    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    description = Column(String(255))

    tours = relationship("Tour", backref='city')


class Tour(Base):
    __tablename__ = 'tours'

    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    description = Column(String(255))
    price = Column(Float, nullable=False)
    duration = Column(Integer)
    start_date = Column(DateTime, nullable=False)
    city_id = Column(Integer, ForeignKey('cities.id'))

    bookings = relationship("Booking", backref='tour')


class Booking(Base):
    __tablename__ = 'bookings'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)  # Replace with actual user model id
    tour_id = Column(Integer, ForeignKey('tours.id'))
    人数 = Column(Integer)  # Number of participants
    date_booked = Column(DateTime, default=datetime.now)


# Create all tables
Base.metadata.create_all(engine)

# Example usage
Session = sessionmaker(bind=engine)
session = Session()

# Add a city
city = City(name="Paris", description="The City of Lights")
session.add(city)

# Add a tour
tour = Tour(name="Eiffel Tower Tour", description="Visit the iconic Eiffel Tower",
            price=50.0, duration=2, start_date=datetime(2024, 3, 1), city=city)
session.add(tour)

# Add a booking
booking = Booking(user_id=1, tour=tour, 人数=2)  # Replace with actual user id
session.add(booking)

# Commit changes
session.commit()

# Query tours in Paris
paris_tours = session.query(Tour).filter_by(city=city).all()

# Close session
session.close()
