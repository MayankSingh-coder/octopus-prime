// AttentionWeightsVisualization Component
// Visualizes the attention weights between tokens

const AttentionWeightsVisualization = ({ attentionData, words }) => {
    const svgRef = React.useRef(null);
    
    React.useEffect(() => {
        if (!attentionData || attentionData.length === 0 || !words || words.length === 0) return;
        
        // Clear previous visualization
        d3.select(svgRef.current).selectAll("*").remove();
        
        // Set up dimensions
        const margin = { top: 30, right: 30, bottom: 30, left: 30 };
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;
        
        // Create SVG
        const svg = d3.select(svgRef.current)
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
        
        // Set up scales
        const xScale = d3.scaleBand()
            .domain(words.map((_, i) => i))
            .range([0, width])
            .padding(0.1);
        
        const yScale = d3.scaleBand()
            .domain(words.map((_, i) => i))
            .range([0, height])
            .padding(0.1);
        
        // Color scale for attention weights
        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, d3.max(attentionData, d => d.weight)]);
        
        // Create heatmap cells
        svg.selectAll(".attention-cell")
            .data(attentionData)
            .enter()
            .append("rect")
            .attr("class", "attention-cell")
            .attr("x", d => xScale(d.source_idx))
            .attr("y", d => yScale(d.target_idx))
            .attr("width", xScale.bandwidth())
            .attr("height", yScale.bandwidth())
            .attr("fill", d => colorScale(d.weight))
            .attr("stroke", "#fff")
            .attr("stroke-width", 0.5)
            .on("mouseover", function(event, d) {
                // Highlight cell on hover
                d3.select(this)
                    .attr("stroke", "#333")
                    .attr("stroke-width", 2);
                
                // Show tooltip
                const tooltip = d3.select(".attention-tooltip");
                tooltip.style("display", "block")
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY + 10}px`)
                    .html(`<strong>From:</strong> ${d.source_word} (${d.source_idx})<br>
                           <strong>To:</strong> ${d.target_word} (${d.target_idx})<br>
                           <strong>Weight:</strong> ${d.weight.toFixed(4)}`);
            })
            .on("mouseout", function() {
                // Reset highlight
                d3.select(this)
                    .attr("stroke", "#fff")
                    .attr("stroke-width", 0.5);
                
                // Hide tooltip
                d3.select(".attention-tooltip")
                    .style("display", "none");
            });
        
        // Add X axis labels (source tokens)
        svg.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale)
                .tickFormat(i => words[i] || ""))
            .selectAll("text")
            .attr("y", 10)
            .attr("x", -5)
            .attr("transform", "rotate(-45)")
            .style("text-anchor", "end");
        
        // Add Y axis labels (target tokens)
        svg.append("g")
            .attr("class", "y-axis")
            .call(d3.axisLeft(yScale)
                .tickFormat(i => words[i] || ""));
        
        // Add title
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", -10)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text("Attention Weights Heatmap");
        
        // Add X axis title
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height + margin.bottom - 5)
            .attr("text-anchor", "middle")
            .text("Source Tokens");
        
        // Add Y axis title
        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2)
            .attr("y", -margin.left + 15)
            .attr("text-anchor", "middle")
            .text("Target Tokens");
        
        // Add color legend
        const legendWidth = 20;
        const legendHeight = height / 2;
        
        // Create gradient for legend
        const defs = svg.append("defs");
        const gradient = defs.append("linearGradient")
            .attr("id", "attention-gradient")
            .attr("x1", "0%")
            .attr("y1", "100%")
            .attr("x2", "0%")
            .attr("y2", "0%");
        
        // Add gradient stops
        gradient.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", colorScale(0));
        
        gradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", colorScale(d3.max(attentionData, d => d.weight)));
        
        // Add legend rectangle
        svg.append("rect")
            .attr("x", width + 10)
            .attr("y", height / 4)
            .attr("width", legendWidth)
            .attr("height", legendHeight)
            .style("fill", "url(#attention-gradient)");
        
        // Add legend axis
        const legendScale = d3.scaleLinear()
            .domain([0, d3.max(attentionData, d => d.weight)])
            .range([legendHeight, 0]);
        
        svg.append("g")
            .attr("transform", `translate(${width + 10 + legendWidth},${height / 4})`)
            .call(d3.axisRight(legendScale)
                .ticks(5)
                .tickFormat(d3.format(".2f")));
        
        // Add legend title
        svg.append("text")
            .attr("x", width + 10 + legendWidth / 2)
            .attr("y", height / 4 - 10)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text("Weight");
        
    }, [attentionData, words]);
    
    return (
        <div className="attention-visualization">
            <div className="visualization-container">
                <svg ref={svgRef}></svg>
                <div className="attention-tooltip" style={{ display: 'none' }}></div>
            </div>
            <div className="attention-info">
                <p>This heatmap shows the attention weights between tokens. Darker colors indicate stronger attention.</p>
                <p>Hover over cells to see detailed weight information.</p>
            </div>
        </div>
    );
};