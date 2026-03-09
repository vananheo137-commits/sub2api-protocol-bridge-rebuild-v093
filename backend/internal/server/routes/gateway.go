package routes

import (
	"net/http"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/handler"
	"github.com/Wei-Shaw/sub2api/internal/server/middleware"
	"github.com/Wei-Shaw/sub2api/internal/service"

	"github.com/gin-gonic/gin"
)

// RegisterGatewayRoutes registers gateway routes for the supported provider
// compatibility layers.
func RegisterGatewayRoutes(
	r *gin.Engine,
	h *handler.Handlers,
	apiKeyAuth middleware.APIKeyAuthMiddleware,
	apiKeyService *service.APIKeyService,
	subscriptionService *service.SubscriptionService,
	opsService *service.OpsService,
	settingService *service.SettingService,
	cfg *config.Config,
) {
	bodyLimit := middleware.RequestBodyLimit(cfg.Gateway.MaxBodySize)
	soraMaxBodySize := cfg.Gateway.SoraMaxBodySize
	if soraMaxBodySize <= 0 {
		soraMaxBodySize = cfg.Gateway.MaxBodySize
	}
	soraBodyLimit := middleware.RequestBodyLimit(soraMaxBodySize)
	clientRequestID := middleware.ClientRequestID()
	opsErrorLogger := handler.OpsErrorLoggerMiddleware(opsService)

	requireGroupAnthropic := middleware.RequireGroupAssignment(settingService, middleware.AnthropicErrorWriter)
	requireGroupGoogle := middleware.RequireGroupAssignment(settingService, middleware.GoogleErrorWriter)

	gateway := r.Group("/v1")
	gateway.Use(bodyLimit)
	gateway.Use(clientRequestID)
	gateway.Use(opsErrorLogger)
	gateway.Use(gin.HandlerFunc(apiKeyAuth))
	gateway.Use(requireGroupAnthropic)
	{
		gateway.POST("/messages", func(c *gin.Context) {
			if getGroupPlatform(c) == service.PlatformOpenAI {
				h.OpenAIGateway.Messages(c)
				return
			}
			h.Gateway.Messages(c)
		})
		gateway.POST("/messages/count_tokens", func(c *gin.Context) {
			if getGroupPlatform(c) == service.PlatformOpenAI {
				c.JSON(http.StatusNotFound, gin.H{
					"type": "error",
					"error": gin.H{
						"type":    "not_found_error",
						"message": "Token counting is not supported for this platform",
					},
				})
				return
			}
			h.Gateway.CountTokens(c)
		})
		gateway.GET("/models", h.Gateway.Models)
		gateway.GET("/usage", h.Gateway.Usage)
		gateway.POST("/responses", h.OpenAIGateway.Responses)
		gateway.POST("/responses/*subpath", h.OpenAIGateway.Responses)
		gateway.GET("/responses", h.OpenAIGateway.ResponsesWebSocket)
		gateway.POST("/completions", h.OpenAIGateway.Completions)
		gateway.POST("/chat/completions", h.OpenAIGateway.ChatCompletions)
	}

	gemini := r.Group("/v1beta")
	gemini.Use(bodyLimit)
	gemini.Use(clientRequestID)
	gemini.Use(opsErrorLogger)
	gemini.Use(middleware.APIKeyAuthWithSubscriptionGoogle(apiKeyService, subscriptionService, cfg))
	gemini.Use(requireGroupGoogle)
	{
		gemini.GET("/models", h.Gateway.GeminiV1BetaListModels)
		gemini.GET("/models/:model", h.Gateway.GeminiV1BetaGetModel)
		gemini.POST("/models/*modelAction", h.Gateway.GeminiV1BetaModels)
	}

	r.POST("/responses", bodyLimit, clientRequestID, opsErrorLogger, gin.HandlerFunc(apiKeyAuth), requireGroupAnthropic, h.OpenAIGateway.Responses)
	r.POST("/responses/*subpath", bodyLimit, clientRequestID, opsErrorLogger, gin.HandlerFunc(apiKeyAuth), requireGroupAnthropic, h.OpenAIGateway.Responses)
	r.GET("/responses", bodyLimit, clientRequestID, opsErrorLogger, gin.HandlerFunc(apiKeyAuth), requireGroupAnthropic, h.OpenAIGateway.ResponsesWebSocket)

	r.GET("/antigravity/models", gin.HandlerFunc(apiKeyAuth), requireGroupAnthropic, h.Gateway.AntigravityModels)

	antigravityV1 := r.Group("/antigravity/v1")
	antigravityV1.Use(bodyLimit)
	antigravityV1.Use(clientRequestID)
	antigravityV1.Use(opsErrorLogger)
	antigravityV1.Use(middleware.ForcePlatform(service.PlatformAntigravity))
	antigravityV1.Use(gin.HandlerFunc(apiKeyAuth))
	antigravityV1.Use(requireGroupAnthropic)
	{
		antigravityV1.POST("/messages", h.Gateway.Messages)
		antigravityV1.POST("/messages/count_tokens", h.Gateway.CountTokens)
		antigravityV1.GET("/models", h.Gateway.AntigravityModels)
		antigravityV1.GET("/usage", h.Gateway.Usage)
	}

	antigravityV1Beta := r.Group("/antigravity/v1beta")
	antigravityV1Beta.Use(bodyLimit)
	antigravityV1Beta.Use(clientRequestID)
	antigravityV1Beta.Use(opsErrorLogger)
	antigravityV1Beta.Use(middleware.ForcePlatform(service.PlatformAntigravity))
	antigravityV1Beta.Use(middleware.APIKeyAuthWithSubscriptionGoogle(apiKeyService, subscriptionService, cfg))
	antigravityV1Beta.Use(requireGroupGoogle)
	{
		antigravityV1Beta.GET("/models", h.Gateway.GeminiV1BetaListModels)
		antigravityV1Beta.GET("/models/:model", h.Gateway.GeminiV1BetaGetModel)
		antigravityV1Beta.POST("/models/*modelAction", h.Gateway.GeminiV1BetaModels)
	}

	soraV1 := r.Group("/sora/v1")
	soraV1.Use(soraBodyLimit)
	soraV1.Use(clientRequestID)
	soraV1.Use(opsErrorLogger)
	soraV1.Use(middleware.ForcePlatform(service.PlatformSora))
	soraV1.Use(gin.HandlerFunc(apiKeyAuth))
	soraV1.Use(requireGroupAnthropic)
	{
		soraV1.POST("/chat/completions", h.SoraGateway.ChatCompletions)
		soraV1.GET("/models", h.Gateway.Models)
	}

	if cfg.Gateway.SoraMediaRequireAPIKey {
		r.GET("/sora/media/*filepath", gin.HandlerFunc(apiKeyAuth), h.SoraGateway.MediaProxy)
	} else {
		r.GET("/sora/media/*filepath", h.SoraGateway.MediaProxy)
	}
	r.GET("/sora/media-signed/*filepath", h.SoraGateway.MediaProxySigned)
}

// getGroupPlatform extracts the group platform from the API Key stored in context.
func getGroupPlatform(c *gin.Context) string {
	apiKey, ok := middleware.GetAPIKeyFromContext(c)
	if !ok || apiKey.Group == nil {
		return ""
	}
	return apiKey.Group.Platform
}
