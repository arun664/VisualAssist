// Fix for Navigation Guidance Alerts
// This script modifies the client.js code to prevent navigation alerts from appearing before navigation starts

document.addEventListener('DOMContentLoaded', () => {
    // Wait for NavigationClient to be initialized
    setTimeout(() => {
        // Get the navigation client instance
        const client = window.navigatorClientInstances?.[0];
        
        if (client) {
            console.log('🛠️ Applying fix for premature navigation alerts...');
            
            // Add a navigationActive flag to control when alerts should be shown
            client.navigationActive = false;
            
            // Store the original handleNavigationGuidance method
            const originalHandleNavigationGuidance = client.handleNavigationGuidance;
            
            // Replace with fixed version that checks if navigation is active
            client.handleNavigationGuidance = function(guidanceData) {
                if (!guidanceData) return;
                
                // Check if this is a test navigation (via URL parameter) or if navigation is actually active
                const isTestMode = window.location.search.includes('test=true');
                if (!this.navigationActive && !isTestMode && guidanceData.direction === 'no_path') {
                    console.log('📊 Suppressing navigation alert - navigation not active yet');
                    return;
                }
                
                // Call the original method for normal processing
                originalHandleNavigationGuidance.call(this, guidanceData);
            };
            
            // Update startStreaming to set navigationActive flag
            const originalStartStreaming = client.startStreaming;
            client.startStreaming = async function() {
                await originalStartStreaming.call(this);
                // Mark navigation as active after streaming starts
                this.navigationActive = true;
                console.log('📊 Navigation marked as active - guidance alerts enabled');
            };
            
            // Update stopStreaming to unset navigationActive flag
            const originalStopStreaming = client.stopStreaming;
            client.stopStreaming = function() {
                originalStopStreaming.call(this);
                // Mark navigation as inactive when streaming stops
                this.navigationActive = false;
                console.log('📊 Navigation marked as inactive - guidance alerts disabled');
                
                // Reset guidance message to default when streaming stops
                const guidanceMessage = document.getElementById('guidanceMessage');
                if (guidanceMessage) {
                    guidanceMessage.className = 'guidance-message';
                    guidanceMessage.innerHTML = 'Start streaming to receive navigation assistance';
                }
            };
            
            console.log('✅ Navigation alert fix applied successfully');
        } else {
            console.error('❌ Could not apply navigation alert fix - client not found');
        }
    }, 500); // Short delay to ensure client is initialized
});