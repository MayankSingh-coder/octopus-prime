// Main App Component
// This is the entry point for the React application

const App = () => {
    const [models, setModels] = React.useState([]);
    const [selectedModel, setSelectedModel] = React.useState(null);
    const [modelInfo, setModelInfo] = React.useState(null);
    const [inputText, setInputText] = React.useState('');
    const [predictions, setPredictions] = React.useState(null);
    const [detailedPrediction, setDetailedPrediction] = React.useState(null);
    const [isTraining, setIsTraining] = React.useState(false);
    const [trainingProgress, setTrainingProgress] = React.useState(null);
    const [trainingId, setTrainingId] = React.useState(null);
    const [trainingText, setTrainingText] = React.useState('');
    const [trainingParams, setTrainingParams] = React.useState({
        model_type: 'attention',
        context_size: 3,
        hidden_layers: '128,64',
        learning_rate: 0.05,
        iterations: 1000,
        attention_dim: 40,
        num_heads: 2,
        attention_dropout: 0.1,
        model_name: 'new_model'
    });
    const [error, setError] = React.useState(null);
    
    // Fetch available models on component mount
    React.useEffect(() => {
        fetchModels();
    }, []);
    
    // Fetch model info when a model is selected
    React.useEffect(() => {
        if (selectedModel) {
            fetchModelInfo(selectedModel);
        }
    }, [selectedModel]);
    
    // Poll for training progress when training is in progress
    React.useEffect(() => {
        let intervalId;
        if (isTraining && trainingId) {
            intervalId = setInterval(() => {
                fetchTrainingProgress(trainingId);
            }, 1000);
        }
        return () => {
            if (intervalId) clearInterval(intervalId);
        };
    }, [isTraining, trainingId]);
    
    // Fetch available models
    const fetchModels = async () => {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            setModels(data.models || []);
        } catch (error) {
            console.error('Error fetching models:', error);
            setError('Failed to fetch models. Please try again.');
        }
    };
    
    // Fetch model information
    const fetchModelInfo = async (modelName) => {
        try {
            const response = await fetch(`/api/model/${modelName}/info`);
            const data = await response.json();
            setModelInfo({
                ...data,
                name: modelName
            });
        } catch (error) {
            console.error('Error fetching model info:', error);
            setError('Failed to fetch model information. Please try again.');
        }
    };
    
    // Fetch training progress
    const fetchTrainingProgress = async (id) => {
        try {
            const response = await fetch(`/api/train/progress/${id}`);
            const data = await response.json();
            setTrainingProgress(data);
            
            // Check if training is complete
            if (data.status === 'completed' || data.status === 'error') {
                setIsTraining(false);
                fetchModels(); // Refresh model list
            }
        } catch (error) {
            console.error('Error fetching training progress:', error);
        }
    };
    
    // Handle model selection
    const handleModelSelect = (event) => {
        setSelectedModel(event.target.value);
        setPredictions(null);
        setDetailedPrediction(null);
    };
    
    // Handle input text change
    const handleInputChange = (event) => {
        setInputText(event.target.value);
    };
    
    // Handle training text change
    const handleTrainingTextChange = (event) => {
        setTrainingText(event.target.value);
    };
    
    // Handle training parameter change
    const handleParamChange = (param, value) => {
        setTrainingParams({
            ...trainingParams,
            [param]: value
        });
    };
    
    // Start model training
    const startTraining = async () => {
        if (!trainingText.trim()) {
            setError('Please provide training text.');
            return;
        }
        
        try {
            setIsTraining(true);
            setError(null);
            
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: trainingText,
                    ...trainingParams
                })
            });
            
            const data = await response.json();
            
            if (data.training_id) {
                setTrainingId(data.training_id);
            } else {
                setIsTraining(false);
                setError('Failed to start training. Please try again.');
            }
        } catch (error) {
            console.error('Error starting training:', error);
            setIsTraining(false);
            setError('Failed to start training. Please try again.');
        }
    };
    
    // Cancel training
    const cancelTraining = async () => {
        if (!trainingId) return;
        
        try {
            await fetch(`/api/train/cancel/${trainingId}`, {
                method: 'POST'
            });
            setIsTraining(false);
        } catch (error) {
            console.error('Error canceling training:', error);
        }
    };
    
    // Get predictions
    const getPredictions = async () => {
        if (!selectedModel || !inputText.trim()) {
            setError('Please select a model and enter input text.');
            return;
        }
        
        try {
            setError(null);
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    context: inputText,
                    model_name: selectedModel,
                    top_n: 5
                })
            });
            
            const data = await response.json();
            setPredictions(data);
            setDetailedPrediction(null);
        } catch (error) {
            console.error('Error getting predictions:', error);
            setError('Failed to get predictions. Please try again.');
        }
    };
    
    // Get detailed prediction with visualization data
    const getDetailedPrediction = async () => {
        if (!selectedModel || !inputText.trim()) {
            setError('Please select a model and enter input text.');
            return;
        }
        
        try {
            setError(null);
            
            const response = await fetch('/api/predict/detailed', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    context: inputText,
                    model_name: selectedModel,
                    top_n: 5
                })
            });
            
            const data = await response.json();
            setDetailedPrediction(data);
        } catch (error) {
            console.error('Error getting detailed prediction:', error);
            setError('Failed to get detailed prediction. Please try again.');
        }
    };
    
    return (
        <div className="app-container">
            <header className="app-header">
                <h1 className="app-title">Multi-Head Attention Visualization</h1>
            </header>
            
            <div className="main-content">
                <div className="sidebar">
                    {/* Model Selection */}
                    <div className="card">
                        <h3 className="card-title">Model Selection</h3>
                        <div className="form-group">
                            <label className="form-label">Select Model:</label>
                            <select 
                                className="form-control" 
                                value={selectedModel || ''} 
                                onChange={handleModelSelect}
                            >
                                <option value="">-- Select a model --</option>
                                {models.map(model => (
                                    <option key={model} value={model}>{model}</option>
                                ))}
                            </select>
                        </div>
                        
                        {/* Display model info if a model is selected */}
                        {modelInfo && <ModelInfoPanel modelInfo={modelInfo} />}
                    </div>
                    
                    {/* Training Section */}
                    <div className="card">
                        <h3 className="card-title">Train Model</h3>
                        
                        <div className="form-group">
                            <label className="form-label">Model Type:</label>
                            <div className="radio-group">
                                <label>
                                    <input 
                                        type="radio" 
                                        name="model_type" 
                                        value="standard" 
                                        checked={trainingParams.model_type === 'standard'} 
                                        onChange={() => handleParamChange('model_type', 'standard')}
                                    />
                                    Standard MLP
                                </label>
                                <label>
                                    <input 
                                        type="radio" 
                                        name="model_type" 
                                        value="attention" 
                                        checked={trainingParams.model_type === 'attention'} 
                                        onChange={() => handleParamChange('model_type', 'attention')}
                                    />
                                    Attention-Enhanced
                                </label>
                            </div>
                        </div>
                        
                        <div className="form-group">
                            <label className="form-label">Context Size:</label>
                            <input 
                                type="number" 
                                className="form-control" 
                                value={trainingParams.context_size} 
                                onChange={(e) => handleParamChange('context_size', parseInt(e.target.value))}
                                min="1"
                                max="10"
                            />
                        </div>
                        
                        <div className="form-group">
                            <label className="form-label">Hidden Layers:</label>
                            <input 
                                type="text" 
                                className="form-control" 
                                value={trainingParams.hidden_layers} 
                                onChange={(e) => handleParamChange('hidden_layers', e.target.value)}
                                placeholder="e.g., 128,64"
                            />
                        </div>
                        
                        <div className="form-group">
                            <label className="form-label">Learning Rate:</label>
                            <input 
                                type="number" 
                                className="form-control" 
                                value={trainingParams.learning_rate} 
                                onChange={(e) => handleParamChange('learning_rate', parseFloat(e.target.value))}
                                step="0.01"
                                min="0.001"
                                max="0.5"
                            />
                        </div>
                        
                        <div className="form-group">
                            <label className="form-label">Iterations:</label>
                            <input 
                                type="number" 
                                className="form-control" 
                                value={trainingParams.iterations} 
                                onChange={(e) => handleParamChange('iterations', parseInt(e.target.value))}
                                min="100"
                                max="10000"
                            />
                        </div>
                        
                        {trainingParams.model_type === 'attention' && (
                            <>
                                <div className="form-group">
                                    <label className="form-label">Attention Dimension:</label>
                                    <input 
                                        type="number" 
                                        className="form-control" 
                                        value={trainingParams.attention_dim} 
                                        onChange={(e) => handleParamChange('attention_dim', parseInt(e.target.value))}
                                        min="10"
                                        max="100"
                                    />
                                </div>
                                
                                <div className="form-group">
                                    <label className="form-label">Attention Heads:</label>
                                    <input 
                                        type="number" 
                                        className="form-control" 
                                        value={trainingParams.num_heads} 
                                        onChange={(e) => handleParamChange('num_heads', parseInt(e.target.value))}
                                        min="1"
                                        max="8"
                                    />
                                </div>
                                
                                <div className="form-group">
                                    <label className="form-label">Attention Dropout:</label>
                                    <input 
                                        type="number" 
                                        className="form-control" 
                                        value={trainingParams.attention_dropout} 
                                        onChange={(e) => handleParamChange('attention_dropout', parseFloat(e.target.value))}
                                        step="0.05"
                                        min="0"
                                        max="0.5"
                                    />
                                </div>
                            </>
                        )}
                        
                        <div className="form-group">
                            <label className="form-label">Model Name:</label>
                            <input 
                                type="text" 
                                className="form-control" 
                                value={trainingParams.model_name} 
                                onChange={(e) => handleParamChange('model_name', e.target.value)}
                            />
                        </div>
                        
                        <div className="form-group">
                            <label className="form-label">Training Text:</label>
                            <textarea 
                                className="form-control" 
                                value={trainingText} 
                                onChange={handleTrainingTextChange}
                                rows="6"
                                placeholder="Enter text for training..."
                            ></textarea>
                        </div>
                        
                        {isTraining && trainingProgress && (
                            <div className="training-progress">
                                <div className="progress-bar">
                                    <div 
                                        className="progress-bar-fill" 
                                        style={{ width: `${trainingProgress.progress}%` }}
                                    ></div>
                                </div>
                                <div className="progress-info">
                                    <span>{trainingProgress.message}</span>
                                    <span>{trainingProgress.progress}%</span>
                                </div>
                            </div>
                        )}
                        
                        <div className="button-group">
                            {!isTraining ? (
                                <button 
                                    className="btn btn-primary" 
                                    onClick={startTraining}
                                >
                                    Train Model
                                </button>
                            ) : (
                                <button 
                                    className="btn btn-danger" 
                                    onClick={cancelTraining}
                                >
                                    Cancel Training
                                </button>
                            )}
                        </div>
                    </div>
                </div>
                
                <div className="main-panel">
                    {/* Inference Section */}
                    <div className="card">
                        <h3 className="card-title">Model Inference</h3>
                        
                        <div className="form-group">
                            <label className="form-label">Input Text:</label>
                            <textarea 
                                className="form-control" 
                                value={inputText} 
                                onChange={handleInputChange}
                                rows="4"
                                placeholder="Enter text for prediction..."
                            ></textarea>
                        </div>
                        
                        <div className="button-group">
                            <button 
                                className="btn btn-primary" 
                                onClick={getPredictions}
                                disabled={!selectedModel}
                            >
                                Get Predictions
                            </button>
                            <button 
                                className="btn btn-secondary" 
                                onClick={getDetailedPrediction}
                                disabled={!selectedModel}
                            >
                                Visualize Prediction Process
                            </button>
                        </div>
                        
                        {/* Display basic predictions */}
                        {predictions && (
                            <div className="predictions-container">
                                <h4>Predictions:</h4>
                                <div className="predictions-list">
                                    {predictions.predictions.map((pred, index) => (
                                        <div key={index} className="prediction-item">
                                            <span className="prediction-word">{pred.word}</span>
                                            <span className="prediction-prob">{(pred.probability * 100).toFixed(2)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                    
                    {/* Visualization Section */}
                    {detailedPrediction && (
                        <div className="visualization-container">
                            {/* Tokenization View */}
                            <div className="card">
                                <h3 className="card-title">Tokenization</h3>
                                <TokenizationView tokenization={detailedPrediction.tokenization} />
                            </div>
                            
                            {/* Attention Weights Visualization (only for attention models) */}
                            {detailedPrediction.is_attention_model && detailedPrediction.attention_weights.length > 0 && (
                                <div className="card">
                                    <h3 className="card-title">Attention Weights</h3>
                                    <AttentionWeightsVisualization 
                                        attentionData={detailedPrediction.attention_weights} 
                                        words={detailedPrediction.tokenization.tokenized_words}
                                    />
                                </div>
                            )}
                            
                            {/* Probability Distribution */}
                            <div className="card">
                                <h3 className="card-title">Prediction Probabilities</h3>
                                <ProbabilityDistribution 
                                    topPredictions={detailedPrediction.top_predictions}
                                    fullDistribution={detailedPrediction.full_distribution}
                                />
                            </div>
                            
                            {/* MLP Layer Visualization */}
                            <div className="card">
                                <h3 className="card-title">Neural Network Architecture</h3>
                                <MLPLayerVisualization 
                                    mlpActivations={detailedPrediction.mlp_activations}
                                    isAttentionModel={detailedPrediction.is_attention_model}
                                />
                            </div>
                            
                            {/* Prediction Steps */}
                            <div className="card">
                                <h3 className="card-title">Prediction Process</h3>
                                <PredictionSteps 
                                    tokenization={detailedPrediction.tokenization}
                                    wordEmbeddings={detailedPrediction.word_embeddings}
                                    isAttentionModel={detailedPrediction.is_attention_model}
                                    attentionWeights={detailedPrediction.attention_weights}
                                    mlpActivations={detailedPrediction.mlp_activations}
                                    topPredictions={detailedPrediction.top_predictions}
                                />
                            </div>
                        </div>
                    )}
                </div>
            </div>
            
            {/* Error display */}
            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}
        </div>
    );
};

// Render the App component to the DOM
ReactDOM.render(<App />, document.getElementById('root'));