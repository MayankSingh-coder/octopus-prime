// PredictionSteps Component
// Visualizes the step-by-step process of prediction in the multi-head attention model

const PredictionSteps = ({ tokenization, wordEmbeddings, isAttentionModel, attentionWeights, mlpActivations, topPredictions }) => {
    if (!tokenization || !topPredictions) return null;
    
    // Define the steps in the prediction process
    const steps = [
        {
            title: "Tokenization",
            description: "Input text is split into tokens",
            data: tokenization ? tokenization.tokenized_words : [],
            render: (data) => (
                <div className="step-content">
                    <div className="token-list">
                        {data.map((token, idx) => (
                            <span key={idx} className="token-chip">{token}</span>
                        ))}
                    </div>
                </div>
            )
        },
        {
            title: "Word Embeddings",
            description: "Tokens are converted to vector representations",
            data: wordEmbeddings || [],
            render: (data) => (
                <div className="step-content">
                    <div className="embedding-list">
                        {data.map((embedding, idx) => (
                            <div key={idx} className="embedding-item">
                                <div className="embedding-word">{embedding.word}</div>
                                <div className="embedding-vector">
                                    <div 
                                        className="vector-bar" 
                                        style={{ width: `${Math.min(100, embedding.embedding_norm * 20)}%` }}
                                        title={`Vector norm: ${embedding.embedding_norm.toFixed(2)}`}
                                    ></div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )
        },
        {
            title: "Attention Mechanism",
            description: "Model calculates attention between tokens",
            data: attentionWeights || [],
            render: (data) => (
                <div className="step-content">
                    {data.length > 0 ? (
                        <div className="attention-summary">
                            <p>Attention calculated across {data.length} token pairs</p>
                            <div className="attention-highlights">
                                {data.slice(0, 3).map((item, idx) => (
                                    <div key={idx} className="attention-highlight">
                                        <span className="source-token">{item.source_word}</span>
                                        <span className="attention-arrow">→</span>
                                        <span className="target-token">{item.target_word}</span>
                                        <span className="attention-weight">{(item.weight * 100).toFixed(1)}%</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <p>{isAttentionModel ? "Attention data is being processed..." : "This model does not use attention mechanisms."}</p>
                    )}
                </div>
            )
        },
        {
            title: "MLP Processing",
            description: "Processed through neural network layers",
            data: mlpActivations || [],
            render: (data) => (
                <div className="step-content">
                    {data.length > 0 ? (
                        <div className="mlp-layers">
                            {data.map((layer, idx) => (
                                <div key={idx} className="mlp-layer">
                                    <div className="layer-name">Layer {layer.layer}</div>
                                    <div className="layer-neurons">
                                        {Array.from({ length: Math.min(5, layer.neurons || 10) }, (_, i) => (
                                            <div 
                                                key={i} 
                                                className="neuron" 
                                                style={{ 
                                                    opacity: 0.3 + (0.7 * (i / 5)),
                                                    backgroundColor: `var(--${i % 2 ? 'secondary' : 'primary'}-color)`
                                                }}
                                                title={`Neuron ${i+1}, Activation: ${layer.activation_norm ? layer.activation_norm.toFixed(4) : 'N/A'}`}
                                            ></div>
                                        ))}
                                        {(layer.neurons || 10) > 5 && <div className="more-neurons">...</div>}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <p>Layer activation data is being processed...</p>
                    )}
                </div>
            )
        },
        {
            title: "Final Prediction",
            description: "Model outputs probability distribution",
            data: topPredictions || [],
            render: (data) => (
                <div className="step-content">
                    <div className="prediction-list">
                        {data.map((pred, idx) => (
                            <div key={idx} className="prediction-item">
                                <div className="prediction-rank">{idx + 1}</div>
                                <div className="prediction-word">"{pred.word}"</div>
                                <div className="prediction-bar-container">
                                    <div 
                                        className="prediction-bar" 
                                        style={{ width: `${pred.probability * 100}%` }}
                                    ></div>
                                </div>
                                <div className="prediction-probability">{(pred.probability * 100).toFixed(2)}%</div>
                            </div>
                        ))}
                    </div>
                </div>
            )
        }
    ];
    
    return (
        <div className="prediction-steps">
            <h3 className="section-title">Prediction Process</h3>
            <div className="step-container">
                {steps.map((step, index) => (
                    <div key={index} className="step-card">
                        <div className="step-number">Step {index + 1}</div>
                        <div className="step-title">{step.title}</div>
                        <div className="step-description">{step.description}</div>
                        {step.render(step.data)}
                    </div>
                ))}
            </div>
        </div>
    );
};