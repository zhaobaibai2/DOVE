const NS = 'http://www.w3.org/2000/svg';

fetch('data/main_results.json')
  .then(response => {
    if (!response.ok) throw new Error(`Results data: ${response.status}`);
    return response.json();
  })
  .then(renderSuccessChart)
  .catch(error => {
    document.querySelector('#success-chart').textContent = error.message;
  });

function renderSuccessChart(data) {
  const root = document.querySelector('#success-chart');
  const width = 1000, height = 360, left = 72, top = 28, plotH = 255;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('aria-hidden', 'true');

  for (let tick = 0; tick <= 10; tick += 2) {
    const y = top + plotH - tick / 10 * plotH;
    add('line', {x1:left, y1:y, x2:width-20, y2:y, stroke:'#d7dde1'});
    addText(left - 12, y + 5, `${tick}/10`, 'end', '#66727e', 13);
  }

  const groupW = (width - left - 30) / data.scenario_order.length;
  const barW = 34;
  data.scenario_order.forEach((scenario, scenarioIndex) => {
    const x0 = left + scenarioIndex * groupW + 25;
    data.methods.forEach((method, methodIndex) => {
      const value = method.scenario_successes[scenarioIndex];
      const x = x0 + methodIndex * (barW + 9);
      const h = value / 10 * plotH;
      add('rect', {x, y:top+plotH-h, width:barW, height:h, fill: method.name === 'DOVE' ? '#216e5a' : '#aab4bc'});
      addText(x + barW/2, top+plotH-h-7, value, 'middle', '#17212b', 12, method.name === 'DOVE' ? '700' : '400');
    });
    addText(x0 + 82, top + plotH + 25, scenario, 'middle', '#17212b', 13);
  });

  data.methods.forEach((method, i) => {
    const x = left + 8 + i * 158;
    add('rect', {x, y:height-18, width:16, height:10, fill:method.name === 'DOVE' ? '#216e5a' : '#aab4bc'});
    addText(x+23, height-9, method.name, 'start', '#4e5a64', 12, method.name === 'DOVE' ? '700' : '400');
  });
  root.replaceChildren(svg);

  function add(name, attrs) {
    const node = document.createElementNS(NS, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    svg.append(node);
  }
  function addText(x, y, text, anchor, fill, size, weight='400') {
    const node = document.createElementNS(NS, 'text');
    Object.entries({x, y, 'text-anchor':anchor, fill, 'font-size':size, 'font-weight':weight, 'font-family':'Arial, sans-serif'}).forEach(([key, value]) => node.setAttribute(key, value));
    node.textContent = text;
    svg.append(node);
  }
}

const dialog = document.querySelector('#lightbox');
const dialogImage = dialog.querySelector('img');
document.querySelectorAll('[data-lightbox]').forEach(button => button.addEventListener('click', () => {
  dialogImage.src = button.dataset.lightbox;
  dialogImage.alt = button.querySelector('img').alt;
  dialog.showModal();
}));
dialog.querySelector('button').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

document.querySelector('#copy-bibtex').addEventListener('click', async event => {
  await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
  event.currentTarget.textContent = 'Copied';
  setTimeout(() => { event.currentTarget.textContent = 'Copy BibTeX'; }, 1600);
});
