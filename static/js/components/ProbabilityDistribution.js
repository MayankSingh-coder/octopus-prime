// ProbabilityDistribution Component
// Visualizes the probability distribution of predicted tokens

const ProbabilityDistribution = ({ topPredictions, fullDistribution }) => {
    const svgRef = React.useRef(null);
    
    // Use the appropriate data source for visualization
    const predictions = fullDistribution || topPredictions || [];
    
    React.useEffect(() => {
        if (!predictions || predictions.length === 0) return;
        
        // Clear previous visualization
        d3.select(svgRef.current).selectAll("*").remove();
        
        // Set up dimensions
        const margin = { top: 30, right: 30, bottom: 70, left: 60 };
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;
        
        // Create SVG
        const svg = d3.select(svgRef.current)
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
        
        // Sort predictions by probability (descending)
        const sortedPredictions = [...predictions].sort((a, b) => b.probability - a.probability);
        
        // Set up scales
        const xScale = d3.scaleBand()
            .domain(sortedPredictions.map(d => d.word))
            .range([0, width])
            .padding(0.2);
        
        const yScale = d3.scaleLinear()
            .domain([0, d3.max(sortedPredictions, d => d.probability)])
            .range([height, 0]);
        
        // Add X axis
        svg.append("g")
            .attr("class", "x-axis")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .attr("transform", "translate(-10,0)rotate(-45)")
            .style("text-anchor", "end");
        
        // Add Y axis
        svg.append("g")
            .attr("class", "y-axis")
            .call(d3.axisLeft(yScale).tickFormat(d3.format(".0%")));
        
        // Add Y axis label
        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 0 - margin.left)
            .attr("x", 0 - (height / 2))
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text("Probability");
        
        // Add bars
        svg.selectAll(".bar")
            .data(sortedPredictions)
            .enter()
            .append("rect")
            .attr("class", "probability-bar")
            .attr("x", d => xScale(d.word))
            .attr("y", d => yScale(d.probability))
            .attr("width", xScale.bandwidth())
            .attr("height", d => height - yScale(d.probability))
            .on("mouseover", function(event, d) {
                // Highlight bar on hover
                d3.select(this).attr("fill", "var(--secondary-color)");
                
                // Show tooltip
                const tooltip = d3.select(".probability-tooltip");
                tooltip.style("display", "block")
                    .style("left", `${event.pageX + 10}px`)
                    .style("top", `${event.pageY + 10}px`)
                    .html(`<strong>Word:</strong> ${d.word}<br>
                           <strong>Probability:</strong> ${(d.probability * 100).toFixed(2)}%`);
            })
            .on("mouseout", function() {
                // Reset highlight
                d3.select(this).attr("fill", "var(--primary-color)");
                
                // Hide tooltip
                d3.select(".probability-tooltip")
                    .style("display", "none");
            });
        
        // Add title
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", 0 - (margin.top / 2))
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text("Probability Distribution of Predicted Words");
        
    }, [predictions]);
    
    return (
        <div className="probability-distribution">
            <div className="visualization-container">
                <svg ref={svgRef}></svg>
                <div className="probability-tooltip" style={{ display: 'none' }}></div>
            </div>
            <div className="probability-info">
                <p>This chart shows the probability distribution of predicted words.</p>
                <p>Hover over bars to see detailed probability information.</p>
            </div>
        </div>
    );
};