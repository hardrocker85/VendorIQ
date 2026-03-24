/**
 * Vendor Price Optimizer - Core Application Logic
 * Combines data management, math functions (linear regression),
 * price optimization algorithm, and DOM/Chart updates.
 */

// State
let dailyRecords = [];
const MIN_DATA_POINTS = 3;
let activeProductId = 'default';
let activeUserKey = null;

function getActiveStorageKey() {
    if (!activeUserKey) return null;
    return `viq_product_data_${activeUserKey}_${activeProductId}`;
}

function loadRecords() {
    const key = getActiveStorageKey();
    if (!key) return [];
    try {
        return JSON.parse(localStorage.getItem(key)) || [];
    } catch (e) {
        return [];
    }
}

function saveRecords() {
    const key = getActiveStorageKey();
    if (!key) return;
    localStorage.setItem(key, JSON.stringify(dailyRecords));
}

function updateProductSubtitle() {
    const subtitle = document.getElementById('product-subtitle');
    if (!subtitle) return;
    const productName = getProductName(activeProductId);
    subtitle.textContent = productName ? `Optimizing: ${productName}` : 'Data-driven profit maximization for street vendors';
}

function getProductName(productId) {
    if (!activeUserKey || !productId) return null;
    const listKey = `viq_products_${activeUserKey}`;
    try {
        const products = JSON.parse(localStorage.getItem(listKey)) || [];
        const match = products.find(p => p.id === productId);
        return match ? match.name : null;
    } catch (e) {
        return null;
    }
}

// --- Math & Core Logic ---

/**
 * Perform simple linear regression to find best fit line: y = mx + c (or Q = aP + b)
 * Equivalent to np.polyfit(x, y, 1)
 * @param {Array} x - Independent variable (e.g., price)
 * @param {Array} y - Dependent variable (e.g., quantity)
 * @returns {Object} { slope: a, intercept: b }
 */
function linearRegression(x, y) {
    const n = x.length;
    if (n === 0) return { slope: 0, intercept: 0 };

    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += (x[i] * y[i]);
        sumXX += (x[i] * x[i]);
    }

    const denominator = (n * sumXX) - (sumX * sumX);
    if (denominator === 0) return { slope: 0, intercept: sumY / n }; // Vertical line edge case

    const slope = ((n * sumXY) - (sumX * sumY)) / denominator;
    const intercept = (sumY - (slope * sumX)) / n;

    return { slope, intercept };
}

/**
 * Compute R^2 for a linear model y = a x + b
 */
function regressionR2(x, y, a, b) {
    const n = x.length;
    if (n === 0) return 0;

    const meanY = y.reduce((sum, v) => sum + v, 0) / n;
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < n; i++) {
        const yHat = a * x[i] + b;
        const diff = y[i] - yHat;
        ssRes += diff * diff;
        const diffTot = y[i] - meanY;
        ssTot += diffTot * diffTot;
    }
    if (ssTot === 0) return 0;
    return 1 - (ssRes / ssTot);
}

function pearsonCorrelation(x, y) {
    const n = x.length;
    if (n === 0) return 0;
    const meanX = x.reduce((s, v) => s + v, 0) / n;
    const meanY = y.reduce((s, v) => s + v, 0) / n;

    let num = 0;
    let denX = 0;
    let denY = 0;
    for (let i = 0; i < n; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
    }
    if (denX === 0 || denY === 0) return 0;
    return num / Math.sqrt(denX * denY);
}

function quadraticRegression(x, y) {
    const n = x.length;
    if (n === 0) return { a: 0, b: 0, c: 0 };

    let sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
    let sumY = 0, sumXY = 0, sumX2Y = 0;

    for (let i = 0; i < n; i++) {
        const xi = x[i];
        const yi = y[i];
        const x2 = xi * xi;
        const x3 = x2 * xi;
        const x4 = x2 * x2;

        sumX += xi;
        sumX2 += x2;
        sumX3 += x3;
        sumX4 += x4;
        sumY += yi;
        sumXY += xi * yi;
        sumX2Y += x2 * yi;
    }

    const det =
        sumX4 * (sumX2 * n - sumX * sumX) -
        sumX3 * (sumX3 * n - sumX * sumX2) +
        sumX2 * (sumX3 * sumX - sumX2 * sumX2);

    if (det === 0) return { a: 0, b: 0, c: sumY / n };

    const detA =
        sumX2Y * (sumX2 * n - sumX * sumX) -
        sumX3 * (sumXY * n - sumX * sumY) +
        sumX2 * (sumXY * sumX - sumX2 * sumY);

    const detB =
        sumX4 * (sumXY * n - sumX * sumY) -
        sumX2Y * (sumX3 * n - sumX * sumX2) +
        sumX2 * (sumX3 * sumY - sumX2 * sumXY);

    const detC =
        sumX4 * (sumX2 * sumY - sumX * sumXY) -
        sumX3 * (sumX3 * sumY - sumX2 * sumXY) +
        sumX2Y * (sumX3 * sumX - sumX2 * sumX2);

    return {
        a: detA / det,
        b: detB / det,
        c: detC / det
    };
}

function standardDeviation(arr) {
    const n = arr.length;
    if (n === 0) return 0;
    const mean = arr.reduce((s, v) => s + v, 0) / n;
    const variance = arr.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
    return Math.sqrt(variance);
}

/**
 * The core price optimization numerical search algorithm
 * @param {Array} records - Array of daily data objects
 * @returns {Object} Contains optimal price, predicted profit, and demand model details
 */
function optimizePrice(records) {
    if (records.length < MIN_DATA_POINTS) return null;

    const cleanRecords = records.filter(r => r.quantity > 0 && r.revenue > 0);
    if (cleanRecords.length < MIN_DATA_POINTS) return null;

    // 1. Derive price from revenue / quantity
    const prices = cleanRecords.map(r => r.revenue / r.quantity);
    const quantities = cleanRecords.map(r => r.quantity);
    const costs = cleanRecords.map(r => r.cost);

    // 2. Check price variation (must vary)
    const minObservedPrice = Math.min(...prices);
    const maxObservedPrice = Math.max(...prices);
    if (minObservedPrice === maxObservedPrice) {
        return {
            stop: true,
            warning: 'Try different prices to learn demand'
        };
    }

    // 3. Quadratic demand model
    let { a, b, c } = quadraticRegression(prices, quantities);
    if (a > 0) {
        a = -Math.abs(a);
    }

    // 4. Flat curve warning
    if (Math.abs(a) < 0.0001 && Math.abs(b) < 0.01) {
        return {
            stop: true,
            warning: 'Not enough variation in data. Recommendation may be unreliable'
        };
    }

    const corr = pearsonCorrelation(prices, quantities);
    const confidence = Math.abs(corr);

    // 4. Cost handling (weighted average unit cost)
    const totalCost = costs.reduce((sum, v) => sum + v, 0);
    const totalQty = quantities.reduce((sum, v) => sum + v, 0);
    const avgCost = totalQty > 0 ? totalCost / totalQty : 0;

    // 5. Always run optimization on observed price range (no extension)
    const minPrice = minObservedPrice;
    const maxPrice = maxObservedPrice;
    const step = 0.5;

    const demand = (p) => (a * p * p) + (b * p) + c;

    let optimalPrice = minPrice;
    let maxProfit = -Infinity;
    let expectedDemand = 0;
    const profitCurveData = [];

    for (let price = minPrice; price <= maxPrice + 1e-9; price += step) {
        const qPred = demand(price);
        if (qPred < 0) continue;
        const profit = (price - avgCost) * qPred;
        profitCurveData.push({ x: price, y: profit });

        if (profit > maxProfit) {
            maxProfit = profit;
            optimalPrice = price;
            expectedDemand = qPred;
        }
    }

    if (profitCurveData.length === 0 || maxProfit === -Infinity) {
        return {
            stop: true,
            warning: 'Try different prices to learn demand'
        };
    }

    // Clamp recommendation to a realistic band around observed average
    const avgObservedPrice = prices.reduce((s, v) => s + v, 0) / prices.length;
    let bestPrice = optimalPrice;
    bestPrice = Math.min(bestPrice, avgObservedPrice * 1.15);
    bestPrice = Math.max(bestPrice, avgObservedPrice * 0.85);
    bestPrice = Math.max(minPrice, Math.min(bestPrice, maxPrice));

    const finalDemand = Math.max(demand(bestPrice), 0);
    const finalProfit = (bestPrice - avgCost) * finalDemand;

    const expectedPrice = bestPrice;
    const askingPrice = expectedPrice * 1.1;

    return {
        optimalPrice: bestPrice,
        expectedPrice,
        askingPrice,
        rawBestPrice: bestPrice,
        currentPrice: avgObservedPrice,
        maxProfit: finalProfit,
        expectedDemand: finalDemand,
        globalAvgCost: avgCost,
        model: { type: 'quadratic', a, b, c },
        confidence,
        profitCurve: profitCurveData
    };
}

// --- UI Interaction & DOM Updates ---

// Chart Instances
let demandChartInst = null;
let profitChartInst = null;
let trendsChartInst = null;

// Helpers to format currency
const formatCurr = (val) => '₹' + Math.round(parseFloat(val));
const formatNum = (val) => Math.round(parseFloat(val)).toString();

function updateUI() {
    updateMetrics();
    updateTable();
    updateRecommendationPanel();
    updateCharts();
}

function updateMetrics() {
    if (dailyRecords.length === 0) {
        document.getElementById('metric-avg-price').innerText = '₹0.00';
        document.getElementById('metric-avg-demand').innerText = '0';
        document.getElementById('metric-avg-cost').innerText = '₹0.00';
        document.getElementById('metric-avg-profit').innerText = '₹0.00';
        return;
    }

    const sumList = arr => arr.reduce((a, b) => a + b, 0);
    const n = dailyRecords.length;

    const avgPrice = sumList(dailyRecords.map(r => r.avgPrice)) / n;
    const avgDemand = sumList(dailyRecords.map(r => r.quantity)) / n;
    const avgCost = sumList(dailyRecords.map(r => r.avgCost)) / n;
    const avgProfit = sumList(dailyRecords.map(r => r.profit)) / n;

    document.getElementById('metric-avg-price').innerText = formatCurr(avgPrice);
    document.getElementById('metric-avg-demand').innerText = Math.round(avgDemand);
    document.getElementById('metric-avg-cost').innerText = formatCurr(avgCost);
    document.getElementById('metric-avg-profit').innerText = formatCurr(avgProfit);
}

function updateTable(animateLast = false) {
    const tbody = document.getElementById('table-body');
    tbody.innerHTML = '';
    
    document.getElementById('record-count').innerText = `${dailyRecords.length} entries`;

    if (dailyRecords.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row" id="empty-row-msg">
                <td colspan="7">No data logged yet. Add your first entry.</td>
            </tr>`;
        return;
    }

    dailyRecords.forEach((record, index) => {
        const tr = document.createElement('tr');
        // Add animation class if it's the latest item and flag is true
        if (animateLast && index === dailyRecords.length - 1) {
            tr.classList.add('row-enter');
        }
        
        tr.innerHTML = `
            <td>${index + 1}</td>
            <td>${formatCurr(record.revenue)}</td>
            <td>${formatCurr(record.cost)}</td>
            <td>${record.quantity}</td>
            <td>${formatCurr(record.avgPrice)}</td>
            <td>${formatCurr(record.avgCost)}</td>
            <td style="color:${record.profit >= 0 ? 'var(--success)' : 'var(--danger)'}">${formatCurr(record.profit)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function updateRecommendationPanel() {
    const needsMsg = document.getElementById('needs-data-msg');
    const optimalResult = document.getElementById('optimal-result');
    const progressBar = document.getElementById('data-progress');
    const progressText = document.getElementById('progress-text');

    const count = dailyRecords.length;
    
    if (count < MIN_DATA_POINTS) {
        needsMsg.classList.remove('hide');
        optimalResult.classList.add('hide');
        
        const pct = (count / MIN_DATA_POINTS) * 100;
        progressBar.style.width = `${pct}%`;
        progressText.innerText = `${count} / ${MIN_DATA_POINTS} entries logged`;
    } else {
        needsMsg.classList.add('hide');
        optimalResult.classList.remove('hide');

        const optResult = optimizePrice(dailyRecords);
        if(!optResult) return;

        if (optResult.stop) {
            document.getElementById('rec-ask-price').innerText = 'Set asking price: ₹--';
            document.getElementById('rec-exp-price').innerText = 'Expected selling price: ₹--';
            document.getElementById('rec-reason').innerText = 'Reason: customers will bargain down';
            const insightBadge = document.getElementById('insight-badge');
            const insightText = document.getElementById('insight-text');
            insightBadge.className = 'insight-badge badge-neutral';
            insightBadge.innerText = 'Low';
            insightText.innerText = optResult.warning;
            document.getElementById('model-equation').innerText = 'Demand Model: Not enough variation';
            return;
        }

        // Current Average price
        const currentAvgPrice = optResult.currentPrice;

        document.getElementById('rec-ask-price').innerText = `Set asking price: ${formatCurr(optResult.askingPrice)}`;
        document.getElementById('rec-exp-price').innerText = `Expected selling price: ${formatCurr(optResult.expectedPrice)}`;
        document.getElementById('rec-reason').innerText = 'Reason: customers will bargain down';
        
        const diff = optResult.optimalPrice - currentAvgPrice;
        const insightBadge = document.getElementById('insight-badge');
        const insightText = document.getElementById('insight-text');

        const confidence = optResult.confidence ?? 0;
        const confidenceLabel = confidence < 0.3 ? "Low confidence" : confidence < 0.6 ? "Medium confidence" : "High confidence";

        let directionText = 'Current price is optimal';
        if (Math.abs(diff) >= 0.5) {
            if (diff > 0) {
                directionText = `Increase price by ${formatCurr(diff)}`;
                insightBadge.className = 'insight-badge badge-up';
                insightBadge.innerText = `+${formatCurr(diff)}`;
            } else {
                directionText = `Decrease price by ${formatCurr(Math.abs(diff))}`;
                insightBadge.className = 'insight-badge badge-down';
                insightBadge.innerText = `-${formatCurr(Math.abs(diff))}`;
            }
        } else {
            insightBadge.className = 'insight-badge badge-neutral';
            insightBadge.innerText = 'Optimal';
        }

        if (confidence < 0.3) {
            insightBadge.className = 'insight-badge badge-neutral';
            insightBadge.innerText = 'Low';
            insightText.innerText = `Low confidence estimate – based on limited/ noisy data. ${directionText}`;
        } else {
            insightText.innerText = directionText;
        }

        const a_fmt = optResult.model ? optResult.model.a.toFixed(6) : '0.000000';
        const b_fmt = optResult.model ? optResult.model.b.toFixed(4) : '0.0000';
        const c_fmt = optResult.model ? optResult.model.c.toFixed(2) : '0.00';
        const modelLabel = optResult.model ? optResult.model.type : 'quadratic';

        document.getElementById('model-equation').innerText = 
            `Demand Model (${modelLabel}): Q = ${a_fmt}P^2 + ${b_fmt}P + ${c_fmt} (${confidenceLabel})`;
    }
}

// Chart Management
const defaultChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    color: '#718096',
    plugins: { legend: { labels: { color: '#2d3748' } } },
    scales: {
        x: { grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#718096' } },
        y: { grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#718096' } }
    }
};

function initCharts() {
    Chart.defaults.font.family = "'Inter', sans-serif";
    
    // 1. Demand Chart (Scatter plot + regression line)
    const ctxDemand = document.getElementById('demandChart').getContext('2d');
    demandChartInst = new Chart(ctxDemand, {
        type: 'scatter',
        data: { datasets: [] },
        options: {
            ...defaultChartOptions,
            plugins: {
                ...defaultChartOptions.plugins,
                title: { display: false }
            },
            scales: {
                ...defaultChartOptions.scales,
                x: { ...defaultChartOptions.scales.x, title: { display: true, text: 'Average Selling Price (₹)', color: '#718096' } },
                y: { ...defaultChartOptions.scales.y, title: { display: true, text: 'Quantity Sold', color: '#718096' } }
            }
        }
    });

    // 2. Profit Simulation Chart
    const ctxProfit = document.getElementById('profitChart').getContext('2d');
    profitChartInst = new Chart(ctxProfit, {
        type: 'line',
        data: { datasets: [] },
        options: {
            ...defaultChartOptions,
            elements: { point: { radius: 0 } }, // hide points for smooth curve
            scales: {
                ...defaultChartOptions.scales,
                x: { ...defaultChartOptions.scales.x, type: 'linear', title: { display: true, text: 'Candidate Price (₹)', color: '#718096' } },
                y: { ...defaultChartOptions.scales.y, title: { display: true, text: 'Predicted Profit (₹)', color: '#718096' } }
            }
        }
    });

    // 3. Historical Trends Chart (Bar/Line combo)
    const ctxTrends = document.getElementById('trendsChart').getContext('2d');
    trendsChartInst = new Chart(ctxTrends, {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            ...defaultChartOptions,
            scales: {
                ...defaultChartOptions.scales,
                x: { ...defaultChartOptions.scales.x, title: { display: true, text: 'Entry', color: '#718096' } },
                y: { ...defaultChartOptions.scales.y, title: { display: true, text: 'Amount (₹) / Qty', color: '#718096' } }
            }
        }
    });
}

function updateCharts() {
    if(!demandChartInst) return;

    if (dailyRecords.length === 0) {
        demandChartInst.data.datasets = []; demandChartInst.update();
        profitChartInst.data.datasets = []; profitChartInst.update();
        trendsChartInst.data.labels = []; trendsChartInst.data.datasets = []; trendsChartInst.update();
        return;
    }

    // Prepare real data points for Demand Chart
    const scatterData = dailyRecords.map(r => ({ x: r.avgPrice, y: r.quantity }));
    
    let demandDatasets = [{
        label: 'Daily Sales',
        data: scatterData,
        backgroundColor: '#1a365d',
        borderColor: '#1a365d',
        pointRadius: 6,
        type: 'scatter'
    }];

    // Update charts based on optimization result
    if (dailyRecords.length >= MIN_DATA_POINTS) {
        const result = optimizePrice(dailyRecords);
        if (result) {
            const minX = Math.min(...scatterData.map(d=>d.x));
            const maxX = Math.max(...scatterData.map(d=>d.x));

            if (result.model) {
                const steps = 24;
                const curve = Array.from({ length: steps }, (_, i) => {
                    const p = minX + (i / (steps - 1)) * (maxX - minX);
                    const y = (result.model.a * p * p) + (result.model.b * p) + result.model.c;
                    return { x: p, y };
                });
                demandDatasets.push({
                    label: 'Demand Trend (Quadratic)',
                    data: curve,
                    type: 'line',
                    borderColor: '#c5a880',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                });
            }

            // Update Profit Chart
            const optPoint = { x: result.optimalPrice, y: result.maxProfit };
            
            profitChartInst.data.datasets = [
                {
                    label: 'Predicted Profit',
                    data: result.profitCurve,
                    borderColor: '#2f855a',
                    backgroundColor: 'rgba(47, 133, 90, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Optimal Point',
                    data: [optPoint],
                    type: 'scatter',
                    backgroundColor: '#c5a880',
                    pointRadius: 8,
                    pointStyle: 'star'
                }
            ];
            profitChartInst.update();
        }
    } else {
        profitChartInst.data.datasets = []; 
        profitChartInst.update();
    }

    demandChartInst.data.datasets = demandDatasets;
    demandChartInst.update();

    // Update Trends Chart
    const labels = dailyRecords.map((_, i) => String(i + 1));
    trendsChartInst.data.labels = labels;
    trendsChartInst.data.datasets = [
        {
            label: 'Revenue',
            data: dailyRecords.map(r => r.revenue),
            backgroundColor: 'rgba(26, 54, 93, 0.5)',
            borderColor: '#1a365d',
            borderWidth: 1,
            type: 'bar'
        },
        {
            label: 'Cost',
            data: dailyRecords.map(r => r.cost),
            backgroundColor: 'rgba(197, 48, 48, 0.5)',
            borderColor: '#c53030',
            borderWidth: 1,
            type: 'bar'
        },
        {
            label: 'Profit',
            data: dailyRecords.map(r => r.profit),
            borderColor: '#2f855a',
            backgroundColor: '#2f855a',
            borderWidth: 2,
            type: 'line',
            tension: 0.3
        }
    ];
    trendsChartInst.update();
}

// --- Event Listeners ---

document.addEventListener('DOMContentLoaded', () => {
    const authed = localStorage.getItem('viq_user');
    if (!authed) {
        window.location.href = 'login.html';
        return;
    }
    activeUserKey = authed;

    const params = new URLSearchParams(window.location.search);
    const urlProduct = params.get('product');
    const storedProduct = localStorage.getItem('viq_active_product');
    activeProductId = urlProduct || storedProduct || 'default';
    updateProductSubtitle();

    dailyRecords = loadRecords();

    initCharts();
    
    // Tab switching logic for charts
    const tabs = document.querySelectorAll('.tab-btn');
    const chartWrappers = document.querySelectorAll('.chart-wrapper');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            chartWrappers.forEach(w => w.classList.add('hide'));
            
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-target');
            document.getElementById(targetId).classList.remove('hide');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Form Submission
    document.getElementById('daily-data-form').addEventListener('submit', (e) => {
        e.preventDefault();
        
        const revenue = parseFloat(document.getElementById('revenue-input').value);
        const cost = parseFloat(document.getElementById('cost-input').value);
        const quantity = parseInt(document.getElementById('quantity-input').value, 10);
        
        // Derived values
        const avgPrice = quantity > 0 ? revenue / quantity : 0;
        const avgCost = quantity > 0 ? cost / quantity : 0;
        const profit = revenue - cost;

        const newRecord = { revenue, cost, quantity, avgPrice, avgCost, profit };
        
        dailyRecords.push(newRecord);
        e.target.reset(); // clear form
        
        saveRecords();
        updateUI();
        updateTable(true); // passed true to animate the new row
    });

    // Clear Data Modal Logic
    const clearBtn = document.getElementById('clear-data-btn');
    const clearModal = document.getElementById('clear-data-modal');
    const cancelClearBtn = document.getElementById('cancel-clear-btn');
    const confirmClearBtn = document.getElementById('confirm-clear-btn');
    const clearCloseBtn = document.getElementById('clear-modal-close');

    const openClearModal = () => {
        clearModal.classList.add('open');
        clearModal.setAttribute('aria-hidden', 'false');
    };

    const closeClearModal = () => {
        clearModal.classList.remove('open');
        clearModal.setAttribute('aria-hidden', 'true');
    };

    if (clearBtn) clearBtn.addEventListener('click', openClearModal);
    if (cancelClearBtn) cancelClearBtn.addEventListener('click', closeClearModal);
    if (clearCloseBtn) clearCloseBtn.addEventListener('click', closeClearModal);

    if (clearModal) {
        clearModal.addEventListener('click', (e) => {
            if (e.target === clearModal) closeClearModal();
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && clearModal && clearModal.classList.contains('open')) {
            closeClearModal();
        }
    });

    if (confirmClearBtn) {
        confirmClearBtn.addEventListener('click', () => {
            dailyRecords = [];
            saveRecords();
            updateUI();
            closeClearModal();
        });
    }

    // Initial render
    updateUI();
});
