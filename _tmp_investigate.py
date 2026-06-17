# tmp investigate script
import sys, os, json
sys.path.insert(0, '/Users/wangzeshang1/MyProjects/zequant')

os.chdir('/Users/wangzeshang1/MyProjects/zequant')

print("Importing...", flush=True)
from live.signals.mss_state import _live_db, _load_all_sub_states

print("Connecting to live DB...", flush=True)
db = _live_db()
print("Connected.", flush=True)

r = db.conn.execute("SELECT total_value FROM daily_performance ORDER BY date DESC LIMIT 1").fetchone()
capital = float(r[0]) if r else 0
print(f'capital(DB): {capital}', flush=True)

state_r = db.conn.execute("SELECT value FROM mss_meta WHERE key='last_known_state'").fetchone()
print(f'state: {state_r[0] if state_r else "None"}', flush=True)

pending = db.conn.execute("SELECT value FROM mss_meta WHERE key='pending_state'").fetchone()
pending_days = db.conn.execute("SELECT value FROM mss_meta WHERE key='pending_days'").fetchone()
print(f'Pending state: {pending[0] if pending else "None"}, days: {pending_days[0] if pending_days else "0"}', flush=True)
print(flush=True)

perf_rows = db.conn.execute("SELECT date, total_value, daily_return, cumulative FROM daily_performance ORDER BY date DESC LIMIT 30").fetchall()
print('=== daily_performance history ===', flush=True)
for row in perf_rows:
    print(f'  {row[0]}: total_value={row[1]:.0f}, daily_return={row[2]:.4f}, cumulative={row[3]:.4f}', flush=True)

print(flush=True)
states = _load_all_sub_states(db)
print('=== sub-strategy states ===', flush=True)
total_used = 0
for name, st in sorted(states.items()):
    holdings = st.get('holdings', {})
    uc = st.get('used_capital', 0)
    total_used += uc
    h_count = len(holdings)
    print(f'{name}:', flush=True)
    print(f'  last_date={st["last_date"]}, last_state={st["last_state"]}', flush=True)
    print(f'  used_capital={uc:.2f}, holdings={h_count}', flush=True)
    sym_costs = []
    for sym, h in holdings.items():
        cost = h['shares'] * h['price']
        sym_costs.append((sym, h['shares'], h['price'], cost))
    sym_costs.sort(key=lambda x: -x[3])
    for sym, shares, price, cost in sym_costs:
        print(f'    {sym}: {shares}shares @ {price:.2f} = {cost:.0f}', flush=True)
    total_cost = sum(c for _, _, _, c in sym_costs)
    print(f'  total holdings cost: {total_cost:.0f} (vs used_capital={uc:.0f})', flush=True)

print(f'\n=== Summary ===', flush=True)
print(f'capital(DB): {capital:.0f}', flush=True)
print(f'total used_capital: {total_used:.0f}', flush=True)
print(f'remain: {capital - total_used:.0f}', flush=True)
db.close()
