DROP TABLE IF EXISTS daily_pnl;

CREATE TABLE daily_pnl (
    trader_id INT,
    pnl_date  DATE,
    pnl       INT
);

INSERT INTO daily_pnl VALUES
(101, '2024-01-01', 500),
(102, '2024-01-01', 800),
(103, '2024-01-01', 300),

(101, '2024-01-02', 700),
(102, '2024-01-02', 200),
(103, '2024-01-02', 900),

(101, '2024-01-03', 400),
(102, '2024-01-03', 600),
(103, '2024-01-03', 500);

with daily_rank as (select trader_id, pnl_date, pnl,
      RANK() OVER (PARTITION BY pnl_date ORDER BY pnl desc) as daily_ranking
      FROM daily_pnl)
    select *,
           LAG(daily_ranking) OVER (PARTITION BY trader_id ORDER BY pnl_date) - daily_ranking as rank_change
    from daily_rank
    order by pnl_date, daily_ranking;

with ranking as (select *,
    LAG(pnl) OVER (PARTITION BY trader_id order by pnl_date) as prev_pnl,
    LEAD(pnl) OVER (PARTITION BY trader_id order by pnl_date) as next_pnl,
    LAG(pnl) OVER (PARTITION BY trader_id order by pnl_date) - pnl as pnl_change,
    LEAD(pnl) OVER (PARTITION BY trader_id order by pnl_date) - pnl as next_pnl_change
    from daily_pnl)
    select *
    from ranking;