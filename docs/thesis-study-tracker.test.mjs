import assert from 'node:assert/strict';
import fs from 'node:fs';

const htmlPath = new URL('./thesis-study-tracker.html', import.meta.url);
const html = fs.readFileSync(htmlPath, 'utf8');

const dataMatch = html.match(
  /<script id="plan-data" type="application\/json">([\s\S]*?)<\/script>/,
);
assert.ok(dataMatch, 'page must embed machine-readable plan data');

const plan = JSON.parse(dataMatch[1]);
assert.equal(plan.days.length, 30, 'plan must cover 30 days');
assert.equal(plan.days[0].date, '2026-08-02');
assert.equal(plan.days.at(-1).date, '2026-08-31');

const expectedDates = Array.from({ length: 30 }, (_, index) => {
  const date = new Date(Date.UTC(2026, 7, 2 + index));
  return date.toISOString().slice(0, 10);
});
assert.deepEqual(
  plan.days.map((day) => day.date),
  expectedDates,
  'dates must be continuous',
);

for (const day of plan.days) {
  assert.equal(day.blocks.length, 4, `${day.date} must have four time blocks`);
  assert.ok(day.tasks.length >= 3, `${day.date} must have at least three tasks`);
  assert.ok(day.output.length > 0, `${day.date} must define a deliverable`);
  assert.ok(day.learning.length > 0, `${day.date} must define learning content`);
}

assert.deepEqual(
  [...new Set(plan.days.map((day) => day.phaseId))],
  ['foundation', 'chapter1', 'chapter2', 'chapter3', 'wrap'],
  'plan must use the five expected phases',
);

for (const view of ['today', 'calendar', 'chapters', 'experiments']) {
  assert.match(html, new RegExp(`data-view="${view}"`), `missing ${view} view`);
}

assert.match(html, /localStorage\.setItem/, 'progress must persist locally');
assert.match(
  html,
  /hoursInput\.addEventListener\('input'/,
  'study hours must save and refresh while the user types',
);
assert.match(
  html,
  /@media \(max-width: 620px\)[\s\S]*?\.tab-button\s*\{[\s\S]*?min-width:\s*0;/,
  'all four view tabs must fit the mobile viewport',
);
assert.match(html, /downloadBackup/, 'page must support exporting a backup');
assert.match(html, /resetProgress/, 'page must support resetting progress');

console.log('thesis-study-tracker checks passed');
