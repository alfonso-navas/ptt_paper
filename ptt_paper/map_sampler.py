from ptt_paper.sampler import PTT

map_sampler: dict[str, type[PTT]] = {
    "default": PTT,
}
