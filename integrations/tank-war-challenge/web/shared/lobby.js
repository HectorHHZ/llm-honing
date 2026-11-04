// Render roster text as text nodes. Never expose or use match capability tokens.
const previous = new WeakMap();
export function renderLobby(root, access) {
    if (!root) return;
    root.hidden = !access?.nickname;
    if (root.hidden) return;
    const queued = access.state === 'queued';
    const signature = JSON.stringify([access.state, access.position, access.nickname, access.capacity, access.lobby]);
    if (previous.get(root)?.signature === signature) return;
    const oldPosition = previous.get(root)?.position;
    const home = document.getElementById('lobby-home');
    const start = document.getElementById('start');
    if (queued) start.before(root);
    else if (root.parentElement !== home) home.append(root);
    root.classList.toggle('waiting-view', queued);
    const {playing = [], waiting = []} = access.lobby;
    root.querySelector('.lobby-count').textContent = `${playing.length} / ${access.capacity} playing`;
    root.querySelector('.lobby-waiting-count').textContent = `${waiting.length} waiting`;
    const players = root.querySelector('.lobby-playing');
    players.replaceChildren(...playing.map(player => {
        const chip = document.createElement('span');
        chip.className = 'lobby-player' + (player.is_you ? ' is-you' : '');
        chip.textContent = player.nickname + (player.is_you ? ' · You' : '');
        return chip;
    }));
    const list = root.querySelector('.lobby-queue'), scroll = list.scrollTop;
    list.replaceChildren(...waiting.map(player => {
        const row = document.createElement('li');
        row.className = player.is_you ? 'is-you' : '';
        if (player.is_you) row.setAttribute('aria-current', 'true');
        const position = document.createElement('span'), name = document.createElement('strong'), you = document.createElement('span');
        position.className = 'lobby-position';position.textContent = `#${player.position}`;
        name.textContent = player.nickname;
        you.className = 'lobby-you';you.textContent = player.is_you ? 'You' : '';
        row.append(position, name, you);return row;
    }));
    root.querySelector('.lobby-empty').hidden = waiting.length > 0;
    list.hidden = waiting.length === 0;
    list.scrollTop = scroll;
    if (queued && oldPosition !== access.position) {
        const own = list.querySelector('[aria-current]');
        if (own) list.scrollTop = Math.max(0, own.offsetTop - list.offsetTop - list.clientHeight / 2 + own.clientHeight / 2);
    }
    previous.set(root, {signature, position: access.position});
}
