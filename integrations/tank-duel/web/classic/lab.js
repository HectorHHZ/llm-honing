const row = (id, cells) => {
    const r = document.createElement("tr");
    for (const text of cells) {
        const c = document.createElement("td");
        c.textContent = text;
        r.append(c);
    }
    document.getElementById(id).append(r);
};
try {
    const res = await fetch("./lab-results.json");
    if (!res.ok) throw Error("Results unavailable");
    const data = await res.json();
    for (const e of data.experiments) {
        row("experiments", [
            e.label,
            e.backend,
            `${e.wins} / ${e.losses} / ${e.draws}`,
            e.status,
        ]);
        const p = document.createElement("p");
        p.className = "reflection";
        p.textContent = `${e.label}: ${e.reflection}`;
        document.getElementById("reflections").append(p);
    }
    for (const c of data.calibration)
        row("latencies", [
            c.label,
            c.serial.p50_ms,
            c.two_seats.p50_ms,
            c.two_seats.p95_ms,
        ]);
    document.getElementById("updated").textContent =
        `Updated ${data.date} · ${data.note}`;
} catch (error) {
    document.getElementById("updated").textContent = error.message;
}
