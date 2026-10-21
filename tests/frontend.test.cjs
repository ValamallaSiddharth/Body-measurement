// Run the actual rendered form script against a minimal test DOM; no browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const display = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

function element() {
  return {
    textContent: '', hidden: true, disabled: false, children: [], style: {}, attributes: {},
    appendChild(child) { this.children.push(child); },
    replaceChildren() { this.children = []; },
    setAttribute(key, value) { this.attributes[key] = value; },
    removeAttribute(key) { delete this.attributes[key]; },
    focus() {},
    get lastChild() { return this.children.at(-1); }
  };
}
const ids = Object.fromEntries(['status', 'results', 'heightValue', 'heightQuality', 'measurements',
  'front_image_error', 'side_image_error', 'gender_error', 'previews'].map(id => [id, element()]));
const button = element();
const inputs = {front_image: {...element(), files: []}, side_image: {...element(), files: []},
  gender: {...element(), value: ''}};
let submit, change, response, requests = 0, release, sentFields;
const form = {
  elements: inputs,
  addEventListener(name, callback) { if (name === 'submit') submit = callback; else if (name === 'change') change = callback; else assert.fail(name); },
  querySelector(selector) { return selector === 'button' ? button : Object.values(inputs).find(i => i.attributes['aria-invalid']); },
  querySelectorAll() { return Object.values(inputs).filter(i => i.attributes['aria-invalid']); }
};
const document = {
  querySelector(selector) { return selector === '#uploadForm' ? form : ids[selector.slice(1)]; },
  querySelectorAll() { return Object.entries(ids).filter(([name]) => name.endsWith('_error')).map(([,value]) => value); },
  getElementById(id) { return ids[id]; },
  createElement: element
};
const event = {preventDefault() {}, target: form};
vm.runInNewContext(source, {document, FormData: class {
  constructor() { this.fields = {}; }
  append(name, value) { this.fields[name] = value; }
}, fetch: async (url, options) => {
  sentFields = options.body.fields;
  requests++;
  await new Promise(resolve => release = resolve);
  return response;
}});

(async () => {
  assert.equal(button.disabled, true);
  await submit(event);
  assert.equal(requests, 0);
  assert.match(ids.front_image_error.textContent, /Upload/);
  assert.match(ids.side_image_error.textContent, /Upload/);
  assert.match(ids.gender_error.textContent, /Select/);
  for (const view of ['front', 'side']) inputs[`${view}_image`].files = [{name: `${view}.png`, type: 'image/png', size: 200}];
  inputs.gender.value = 'male';
  change();
  assert.equal(button.disabled, false);
  response = {ok: true, json: async () => ({display, inferred_height_cm: display.height.cm, inferred_height_in: display.height.inches,
    calibrated_photos: {front: {url: '/previews/test/front_keypoints.png', pixels_per_cm: 4}, side: {url: '/previews/test/side_keypoints.png', pixels_per_cm: 4}}})};
  const pending = submit(event);
  assert.equal(button.disabled, true);
  assert.equal(button.textContent, 'Processing…');
  assert.equal(ids.results.hidden, true);
  release();
  await pending;
  assert.equal(button.disabled, false);
  assert.equal(button.textContent, 'Fit and measure');
  assert.equal(ids.results.hidden, false);
  assert.deepEqual(Object.keys(sentFields).sort(), ['front_image', 'gender', 'side_image']);
  assert.equal(ids.measurements.children.length, 9);
  assert.equal(ids.previews.children.length, 2);
  assert.equal(ids.previews.children[0].children[0].src, '/previews/test/front_keypoints.png');
  assert.match(ids.heightValue.textContent, /cm \/ .* inches/);
  assert.equal(ids.measurements.children[0].children[1].textContent, display.rows[0].cm.toFixed(1));
  assert.equal(ids.measurements.children[0].children[2].textContent, display.rows[0].inches.toFixed(1));
  assert.equal(ids.measurements.children[0].lastChild.textContent, display.rows[0].status);

  response = {ok: false, json: async () => ({error: 'Body not fully visible. Please provide a new capture.'})};
  const rejected = submit(event);
  release();
  await rejected;
  assert.equal(ids.results.hidden, true);
  assert.match(ids.status.textContent, /new capture/);
  assert.equal(button.disabled, false);

  inputs.front_image.files[0].name = 'front.gif';
  await submit(event);
  assert.equal(requests, 2);
  assert.match(ids.front_image_error.textContent, /JPG or PNG/);
  console.log('Frontend validation, loading, height/table rendering and rejection checks passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
