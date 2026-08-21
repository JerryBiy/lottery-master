module.exports = {
  // 上线前替换为已在微信公众平台配置的 HTTPS 业务域名。
  API_BASE_URL: "http://127.0.0.1:5000",
  // 正式环境配置 WECHAT_APP_SECRET 后改为 true；AppSecret 只保存在后端。
  WECHAT_AUTH_ENABLED: false,
};
