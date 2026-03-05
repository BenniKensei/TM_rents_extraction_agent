Select *
from timisoara_rents;
SELECT neighborhood,
    COUNT(*) as total_listings,
    ROUND(AVG(monthly_rent_eur), 0) as avg_rent_eur,
    ROUND(AVG(rooms), 1) as avg_rooms
FROM timisoara_rents
GROUP BY neighborhood
HAVING COUNT(*) > 1
ORDER BY avg_rent_eur DESC;
DROP TABLE timisoara_rents;