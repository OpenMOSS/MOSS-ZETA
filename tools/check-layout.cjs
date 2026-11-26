const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
let checked = 0;
for (const directory of ['harness/src', 'harness/scripts', 'harness/tests', 'cadence/src', 'cadence/scripts', 'website', 'tools']) {
  for (const name of fs.readdirSync(path.join(root, directory))) {
    if (!/\.(js|cjs)$/.test(name)) continue;
    const file = path.join(root, directory, name);
    const syntax = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
    if (syntax.status) throw new Error(syntax.stderr);
    for (const match of fs.readFileSync(file, 'utf8').matchAll(/require\(['"](\.[^'"]+)['"]\)/g)) {
      const target = path.resolve(path.dirname(file), match[1]);
      if (![target, target + '.js', target + '.cjs', path.join(target, 'index.js')].some(p => fs.existsSync(p)))
        throw new Error(`Missing import in ${path.relative(root, file)}: ${match[1]}`);
    }
    checked++;
  }
}
for (const name of ['fixedSubmissionClock', 'streamCadenceExecutor', 'highFrequencyActionParser', 'policyActionStream', 'actionExecutor']) {
  if (fs.existsSync(path.join(root, 'harness/src', name + '.js'))) throw new Error('Duplicate shared component: ' + name);
}
for (const name of ['gamepad_stream_scheduler.py', 'observation-clock.py']) {
  if (fs.existsSync(path.join(root, 'harness/scripts', name))) throw new Error('Duplicate shared scheduler: ' + name);
}
const pkg = require('../package.json');
for (const name of pkg.workspaces) {
  if (require(path.join(root, name, 'package.json')).version !== pkg.version) throw new Error('Workspace version mismatch: ' + name);
}
console.log(`Checked ${checked} JavaScript files, relative imports, shared-component ownership and workspace versions.`);
