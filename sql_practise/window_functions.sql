CREATE TABLE order_events (
    event_id BIGINT PRIMARY KEY,
    order_id INT NOT NULL,
    event_time TIMESTAMP NOT NULL,
    status VARCHAR(10) NOT NULL,
    quantity INT NOT NULL
);

-- Optimization: This index allows the database to quickly group 
-- by order_id and sort by time without reading the entire table.
CREATE INDEX idx_order_time ON order_events (order_id, event_time DESC);

INSERT INTO order_events (event_id, order_id, event_time, status, quantity) VALUES
(1, 101, '2026-05-25 10:00:00', 'NEW', 100),
(2, 101, '2026-05-25 10:00:05', 'PARTIAL', 50),
(3, 102, '2026-05-25 10:00:00', 'NEW', 200),
-- Edge Case: Identical time for order 102, conflict resolution required
(4, 102, '2026-05-25 10:00:00', 'FILLED', 200), 
-- Edge Case: Out of order arrival, time is earlier than previous event
(5, 101, '2026-05-25 09:59:00', 'NEW', 100), 
-- Edge Case: Later event arriving
(6, 101, '2026-05-25 10:00:10', 'FILLED', 50);

select order_id, 
    status, 
    quantity, 
    event_time

from (select 
    order_id,
    status,
    quantity,
    event_time,
    row_number() over (partition by order_id order by event_time desc, -- latest time is filled with 1
    CASE status 
                    WHEN 'FILLED' THEN 1 
                    WHEN 'CANCELLED' THEN 2 
                    WHEN 'PARTIAL' THEN 3 
                    WHEN 'NEW' THEN 4 
                    ELSE 5 
                END ASC) as rn --priority given to fill but if not exists moves to the next
    from order_events)

as final
where rn = 1;