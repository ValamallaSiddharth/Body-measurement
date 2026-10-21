// Dependency-free behavioral checks for the actual inline page script.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function setup(fetchImpl, protocol = 'http:') {
  const elements = new Map();
  function element(id) {
    return { id, hidden: false, disabled: false, textContent: '', children: [], files: [],
      handlers: {}, attrs: {}, naturalWidth: 400, naturalHeight: 800,
      addEventListener(name, fn) { this.handlers[name] = fn; },
      append(child) { this.children.push(child); },
      replaceChildren() { this.children = []; },
      setAttribute(key, value) { this.attrs[key] = value; },
      removeAttribute(key) { delete this.attrs[key]; },
      async decode() {}
    };
  }
  const document = {
    getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
    createElement(tag) { return element(tag); }
  };
  let timeoutCallback, cleared = false;
  const context = {
    document, window: {}, location: { protocol }, fetch: fetchImpl,
    AbortController, FormData: class {}, URL: { createObjectURL: () => 'blob:test', revokeObjectURL() {} },
    setTimeout(callback) { timeoutCallback = callback; return 1; },
    clearTimeout() { cleared = true; }
  };
  vm.runInNewContext(script, context, { filename: 'index.html inline script' });
  for (const view of ['front', 'side']) document.getElementById(view).files = [{ name: `${view}.jpg`, size: 1000 }];
  return { get: id => document.getElementById(id), timeout: () => timeoutCallback(),
    cleared: () => cleared,
    submit: () => document.getElementById('measure-form').handlers.submit({ preventDefault() {} }) };
}

const result = {
  status: 'ok', height: { cm: 175, in: 68.9 }, confidence: 'low',
  measurements: {
    head_neck: { neck: { cm: 35, in: 13.8 } },
    torso: { chest: { cm: 90, in: 35.4 } },
    arms: { wrist: { cm: null, in: null } }, legs: {}
  }, warnings: ['<script>must be rendered as text</script>']
};
const reply = (ok, status, body) => ({ ok, status, text: async () => body });

(async () => {
  let release;
  const ui = setup((url, request) => {
    assert.equal(url, '/measure');
    assert.equal(request.method, 'POST');
    assert.ok(request.signal);
    return new Promise(resolve => { release = resolve; });
  });
  const pending = ui.submit();
  assert.equal(ui.get('measure').disabled, true);
  assert.equal(ui.get('progress').hidden, false);
  release(reply(true, 200, JSON.stringify(result)));
  await pending;
  assert.equal(ui.get('results').hidden, false);
  assert.equal(ui.get('height').textContent, '175.0 cm / 68.9 in');
  assert.equal(ui.get('measurement-rows').children.length, 14);
  assert.equal(ui.get('warning-list').children[0].textContent, result.warnings[0]);
  assert.equal(ui.get('measure').disabled, false);
  assert.equal(ui.get('progress').hidden, true);
  assert.ok(ui.cleared());

  const scenarios = [
    [async () => { throw new TypeError('Failed to fetch'); }, /Cannot reach/],
    [async () => reply(false, 422, JSON.stringify({ error: 'No person detected in front image.' })), /No person/],
    [async () => reply(false, 502, '<html>proxy error</html>'), /HTTP 502/],
    [async () => reply(true, 200, '<html>wrong server</html>'), /non-JSON/],
    [async () => reply(true, 200, '{}'), /unexpected result/]
  ];
  for (const [fetchImpl, message] of scenarios) {
    const failed = setup(fetchImpl);
    await failed.submit();
    assert.match(failed.get('error').textContent, message);
    assert.equal(failed.get('error').hidden, false);
    assert.equal(failed.get('measure').disabled, false);
    assert.equal(failed.get('progress').hidden, true);
    assert.equal(failed.get('results').hidden, true);
    assert.ok(failed.cleared());
  }
  const timed = setup((url, options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => {
      const error = new Error('abort'); error.name = 'AbortError'; reject(error);
    });
  }));
  const timedRequest = timed.submit(); timed.timeout(); await timedRequest;
  assert.match(timed.get('error').textContent, /timed out/);
  assert.equal(timed.get('measure').disabled, false);

  const local = setup(() => assert.fail('file:// should be rejected'), 'file:');
  await local.submit(); assert.match(local.get('error').textContent, /http:\/\/127.0.0.1:5000/);

  const preview = setup(() => {});
  const input = preview.get('front');
  await input.handlers.change({ target: input });
  assert.equal(preview.get('front-preview').src, 'blob:test');
  assert.equal(preview.get('front-preview').hidden, false);
  input.files = [{ name: 'bad.exe', size: 200 }];
  await input.handlers.change({ target: input });
  assert.match(preview.get('error').textContent, /JPG, PNG or WebP/);
  assert.equal(input.value, '');
  console.log('Frontend: 10 scenarios passed (success, 5 error responses, timeout, file origin, valid/invalid previews).');
})().catch(error => { console.error(error); process.exitCode = 1; });
