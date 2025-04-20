// MLPLayerVisualization Component
// Visualizes the neural network layers and activations during prediction

const MLPLayerVisualization = ({ mlpActivations, isAttentionModel }) => {
    const svgRef = React.useRef(null);
    
    React.useEffect(() => {
        if (!mlpActivations || mlpActivations.length === 0) return;
        
        // Clear previous visualization
        d3.select(svgRef.current).selectAll("*").remove();
        
        // Set up dimensions
        const margin = { top: 40, right: 30, bottom: 40, left: 50 };
        const width = 600 - margin.left - margin.right;
        const height = 400 - margin.top - margin.bottom;
        
        // Create SVG
        const svg = d3.select(svgRef.current)
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);
        
        // Calculate layer positions
        const layerWidth = width / (mlpActivations.length + 1);
        
        // Find max neurons count for scaling
        const maxNeurons = d3.max(mlpActivations, d => d.neurons || 10);
        
        // Draw connections between layers
        for (let i = 0; i < mlpActivations.length - 1; i++) {
            const sourceLayer = mlpActivations[i];
            const targetLayer = mlpActivations[i + 1];
            
            const sourceNeurons = Math.min(sourceLayer.neurons || 10, 8); // Limit to 8 neurons for visualization
            const targetNeurons = Math.min(targetLayer.neurons || 10, 8);
            
            const sourceX = (i + 1) * layerWidth;
            const targetX = (i + 2) * layerWidth;
            
            // Draw connections between neurons
            for (let s = 0; s < sourceNeurons; s++) {
                const sourceY = (height / (sourceNeurons + 1)) * (s + 1);
                
                for (let t = 0; t < targetNeurons; t++) {
                    const targetY = (height / (targetNeurons + 1)) * (t + 1);
                    
                    // Calculate connection opacity based on layer activations
                    const connectionOpacity = 0.1 + (0.2 * (sourceLayer.activation_norm / 10));
                    
                    svg.append("line")
                        .attr("x1", sourceX)
                        .attr("y1", sourceY)
                        .attr("x2", targetX)
                        .attr("y2", targetY)
                        .attr("stroke", "#aaa")
                        .attr("stroke-width", 0.5)
                        .attr("opacity", connectionOpacity);
                }
            }
        }
        
        // Draw layers and neurons
        mlpActivations.forEach((layer, i) => {
            const layerX = (i + 1) * layerWidth;
            const neurons = Math.min(layer.neurons || 10, 8); // Limit to 8 neurons for visualization
            
            // Draw layer label
            svg.append("text")
                .attr("x", layerX)
                .attr("y", height + 25)
                .attr("text-anchor", "middle")
                .style("font-size", "12px")
                .text(`Layer ${layer.layer}`);
            
            // Draw neurons
            for (let n = 0; n < neurons; n++) {
                const neuronY = (height / (neurons + 1)) * (n + 1);
                
                // Calculate neuron color based on activation
                const activationIntensity = layer.activation_norm ? 
                    Math.min(1, layer.activation_norm / 10) : 0.5;
                
                const neuronColor = d3.interpolateBlues(0.3 + (0.7 * activationIntensity));
                
                // Draw neuron
                svg.append("circle")
                    .attr("cx", layerX)
                    .attr("cy", neuronY)
                    .attr("r", 10)
                    .attr("fill", neuronColor)
                    .attr("stroke", "#333")
                    .attr("stroke-width", 1)
                    .on("mouseover", function(event) {
                        // Highlight neuron on hover
                        d3.select(this)
                            .attr("stroke", "var(--secondary-color)")
                            .attr("stroke-width", 2);
                        
                        // Show tooltip
                        const tooltip = d3.select(".mlp-tooltip");
                        tooltip.style("display", "block")
                            .style("left", `${event.pageX + 10}px`)
                            .style("top", `${event.pageY + 10}px`)
                            .html(`<strong>Layer ${layer.layer}, Neuron ${n+1}</strong><br>
                                   Activation: ${layer.activation_norm ? layer.activation_norm.toFixed(4) : 'N/A'}`);
                    })
                    .on("mouseout", function() {
                        // Reset highlight
                        d3.select(this)
                            .attr("stroke", "#333")
                            .attr("stroke-width", 1);
                        
                        // Hide tooltip
                        d3.select(".mlp-tooltip")
                            .style("display", "none");
                    });
            }
            
            // If there are more neurons than we're showing, add an ellipsis
            if (layer.neurons > 8) {
                svg.append("text")
                    .attr("x", layerX)
                    .attr("y", height - 20)
                    .attr("text-anchor", "middle")
                    .style("font-size", "16px")
                    .text("...");
            }
        });
        
        // Add input layer label
        svg.append("text")
            .attr("x", 0)
            .attr("y", height / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text("Input");
        
        // Add output layer label
        svg.append("text")
            .attr("x", width)
            .attr("y", height / 2)
            .attr("text-anchor", "middle")
            .style("font-size", "12px")
            .text("Output");
        
        // Add title
        svg.append("text")
            .attr("x", width / 2)
            .attr("y", -20)
            .attr("text-anchor", "middle")
            .style("font-size", "16px")
            .text(`${isAttentionModel ? 'Attention-Enhanced' : 'Standard'} MLP Architecture`);
        
    }, [mlpActivations, isAttentionModel]);
    
    return (
        <div className="mlp-layer-visualization">
            <div className="visualization-container">
                <svg ref={svgRef}></svg>
                <div className="mlp-tooltip" style={{ display: 'none' }}></div>
            </div>
            <div className="mlp-info">
                <p>This visualization shows the neural network architecture and activations during prediction.</p>
                <p>Darker neurons indicate higher activation values. Hover over neurons to see detailed information.</p>
                {isAttentionModel && (
                    <p>The model uses attention mechanisms to focus on relevant parts of the input before processing through the MLP layers.</p>
                )}
            </div>
        </div>
    );
};