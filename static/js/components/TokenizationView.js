// TokenizationView Component
// Displays the tokenization process of the input text

const TokenizationView = ({ tokenization }) => {
    if (!tokenization) return null;
    
    const { original_text, tokenized_words, token_count } = tokenization;
    
    // Create spans for each token with alternating colors for better visualization
    const renderTokenizedText = () => {
        let result = [];
        let currentPosition = 0;
        
        // Find each token in the original text and highlight it
        tokenized_words.forEach((token, index) => {
            // Find the token in the remaining text
            const tokenPosition = original_text.indexOf(token, currentPosition);
            
            if (tokenPosition >= 0) {
                // Add any text before the token
                if (tokenPosition > currentPosition) {
                    result.push(
                        <span key={`pre-${index}`} className="non-token-text">
                            {original_text.substring(currentPosition, tokenPosition)}
                        </span>
                    );
                }
                
                // Add the token with highlighting
                result.push(
                    <span 
                        key={`token-${index}`} 
                        className={`token token-${index % 2}`}
                        title={`Token ${index + 1}: ${token}`}
                    >
                        {token}
                    </span>
                );
                
                // Update current position
                currentPosition = tokenPosition + token.length;
            }
        });
        
        // Add any remaining text
        if (currentPosition < original_text.length) {
            result.push(
                <span key="remaining" className="non-token-text">
                    {original_text.substring(currentPosition)}
                </span>
            );
        }
        
        return result;
    };
    
    return (
        <div className="tokenization-view">
            <div className="tokenization-stats">
                <div className="stat-item">
                    <span className="stat-label">Total Tokens:</span>
                    <span className="stat-value">{token_count}</span>
                </div>
            </div>
            
            <div className="tokenization-original">
                <h4>Original Text:</h4>
                <div className="text-content">{original_text}</div>
            </div>
            
            <div className="tokenization-result">
                <h4>Tokenized Text:</h4>
                <div className="text-content">{renderTokenizedText()}</div>
            </div>
            
            <div className="token-list">
                <h4>Token List:</h4>
                <div className="token-grid">
                    {tokenized_words.map((token, index) => (
                        <div key={index} className={`token-item token-${index % 2}`}>
                            <span className="token-index">{index + 1}</span>
                            <span className="token-text">{token}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};