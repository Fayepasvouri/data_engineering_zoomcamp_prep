DROP TABLE IF EXISTS trades;

CREATE TABLE trades (
    trade_id   INTEGER,
    trader_id  INTEGER,
    pnl        INTEGER,
    trade_date TEXT
);

INSERT INTO trades (trade_id, trader_id, pnl, trade_date) VALUES
(1, 101, 500, '2024-01-01'),
(2, 101, 300, '2024-01-02'),
(3, 101, 700, '2024-01-03'),
(4, 102, 1000, '2024-01-01'),
(5, 102, 800, '2024-01-02'),
(6, 103, 400, '2024-01-01'),
(7, 104, 600, '2024-01-01'),
(8, 104, 600, '2024-01-02'),
(9, 104, 200, '2024-01-03');


-- Query to find the second highest PnL for each trader

WITH ranking as 
   (SELECT trader_id, pnl, trade_date,
    ROW_NUMBER() OVER 
    (PARTITION BY trader_id ORDER by pnl DESC) as pnl_rank
    FROM trades)
select *
from ranking
where pnl_rank = 2;