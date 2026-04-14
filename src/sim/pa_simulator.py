def build_pre_pa_row(
    batter_row: pd.Series,
    pitcher_row: pd.Series,
    state: GameState,
) -> dict[str, Any]:
    row: dict[str, Any] = {}

    row["inning"] = state.inning
    row["outs_before_pa"] = state.outs
    row["inning_topbot"] = state.half
    row["base_state_before"] = encode_base_state(state.on_1b, state.on_2b, state.on_3b)
    row["on_1b"] = state.on_1b
    row["on_2b"] = state.on_2b
    row["on_3b"] = state.on_3b
    row["base_runner_count"] = state.on_1b + state.on_2b + state.on_3b
    row["risp_flag"] = int(state.on_2b == 1 or state.on_3b == 1)
    row["bases_empty_flag"] = int((state.on_1b + state.on_2b + state.on_3b) == 0)
    row["is_top_inning"] = int(state.half == "TOP")
    row["is_bot_inning"] = int(state.half == "BOT")
    row["two_out_flag"] = int(state.outs == 2)

    if state.inning <= 3:
        row["inning_bucket"] = "early"
    elif state.inning <= 6:
        row["inning_bucket"] = "mid"
    else:
        row["inning_bucket"] = "late"

    row["batter_id"] = batter_row.get("batter_id")
    row["pitcher_id"] = pitcher_row.get("pitcher_id")
    row["lineup_slot"] = batter_row.get("lineup_slot")

    for k, v in batter_row.items():
        row[k] = v
    for k, v in pitcher_row.items():
        row[k] = v

    def get_num(key: str) -> float:
        val = row.get(key, np.nan)
        return float(val) if pd.notna(val) else np.nan

    pairs_diff = [
        ("matchup_hit_rate_diff", "bat_hit_rate_roll30", "pit_hit_rate_roll30"),
        ("matchup_hr_rate_diff", "bat_hr_rate_roll30", "pit_hr_rate_roll30"),
        ("matchup_bb_rate_diff", "bat_bb_rate_roll30", "pit_bb_rate_roll30"),
        ("matchup_k_pressure_diff", "bat_so_rate_roll30", "pit_k_rate_roll30"),
        ("matchup_contact_diff", "bat_contact_rate_roll30", "pit_contact_rate_roll30"),
        ("matchup_whiff_diff", "bat_whiff_rate_roll30", "pit_whiff_rate_roll30"),
        ("matchup_hard_hit_diff", "bat_hard_hit_rate_roll30", "pit_hard_hit_rate_roll30"),
        ("matchup_barrel_diff", "bat_barrel_rate_roll30", "pit_barrel_rate_roll30"),
        ("matchup_power_diff", "bat_iso_roll30", "pit_hr_rate_roll30"),
    ]
    for new_col, a, b in pairs_diff:
        av = get_num(a)
        bv = get_num(b)
        row[new_col] = av - bv if pd.notna(av) and pd.notna(bv) else np.nan

    pairs_x = [
        ("matchup_hr_pressure_x", "bat_hr_rate_roll30", "pit_hr_rate_roll30"),
        ("matchup_hit_pressure_x", "bat_hit_rate_roll30", "pit_hit_rate_roll30"),
        ("matchup_walk_pressure_x", "bat_bb_rate_roll30", "pit_bb_rate_roll30"),
        ("matchup_k_pressure_x", "bat_so_rate_roll30", "pit_k_rate_roll30"),
        ("matchup_contact_x", "bat_contact_rate_roll30", "pit_contact_rate_roll30"),
        ("matchup_hard_hit_x", "bat_hard_hit_rate_roll30", "pit_hard_hit_rate_roll30"),
        ("matchup_barrel_x", "bat_barrel_rate_roll30", "pit_barrel_rate_roll30"),
    ]
    for new_col, a, b in pairs_x:
        av = get_num(a)
        bv = get_num(b)
        row[new_col] = av * bv if pd.notna(av) and pd.notna(bv) else np.nan

    return row
