// ModelInfoPanel Component
// Displays information about the model being used

const ModelInfoPanel = ({ modelInfo }) => {
    if (!modelInfo) return null;
    
    return (
        <div className="card">
            <h3 className="card-title">Model Information</h3>
            <div className="model-info-grid">
                <div className="model-info-item">
                    <div className="model-info-label">Model Name</div>
                    <div className="model-info-value">{modelInfo.name}</div>
                </div>
                
                <div className="model-info-item">
                    <div className="model-info-label">Model Type</div>
                    <div className="model-info-value">{modelInfo.type === 'attention' ? 'Attention-Enhanced MLP' : 'Standard MLP'}</div>
                </div>
                
                <div className="model-info-item">
                    <div className="model-info-label">Context Size</div>
                    <div className="model-info-value">{modelInfo.context_size}</div>
                </div>
                
                <div className="model-info-item">
                    <div className="model-info-label">Embedding Dimension</div>
                    <div className="model-info-value">{modelInfo.embedding_dim}</div>
                </div>
                
                <div className="model-info-item">
                    <div className="model-info-label">Vocabulary Size</div>
                    <div className="model-info-value">{modelInfo.vocabulary_size.toLocaleString()}</div>
                </div>
                
                {modelInfo.type === 'attention' && (
                    <>
                        <div className="model-info-item">
                            <div className="model-info-label">Attention Dimension</div>
                            <div className="model-info-value">{modelInfo.attention_dim}</div>
                        </div>
                        
                        <div className="model-info-item">
                            <div className="model-info-label">Attention Heads</div>
                            <div className="model-info-value">{modelInfo.num_attention_heads}</div>
                        </div>
                        
                        <div className="model-info-item">
                            <div className="model-info-label">Attention Dropout</div>
                            <div className="model-info-value">{modelInfo.attention_dropout}</div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};