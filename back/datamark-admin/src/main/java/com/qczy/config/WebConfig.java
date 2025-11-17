package com.qczy.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.servlet.config.annotation.PathMatchConfigurer;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.springframework.web.util.UrlPathHelper;

import java.nio.charset.StandardCharsets;

/**
 * @BelongsProject: demo
 * @Author: DanBrown
 * @CreateTime: 2020-03-28 14:33
 * @description: TODO
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Value("${upload.tempPath}")
    private String uploadTempPath;

    @Value("${upload.port}")
    private String filePort;
    @Value("${upload.formalPath}")
    private String formalPath;
    @Value("${upload.modelDescPath}")
    private String modelDescPath;
    @Value("${upload.modelCasePath}")
    private String modelCasePath;


    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {


/*
            registry.addResourceHandler("swagger-ui.html")
                    .addResourceLocations("classpath:/META-INF/resources/");
            registry.addResourceHandler("/webjars/**")
                    .addResourceLocations("classpath:/META-INF/resources/webjars/");*/


        //  /home/file/**为前端URL访问路径  后面 file:xxxx为本地磁盘映射 file:/C:/aaa/bbb/
        registry.addResourceHandler("/temp/file/**").addResourceLocations("file:" + uploadTempPath);
        registry.addResourceHandler("/formal/**").addResourceLocations("file:" + formalPath);
        // registry.addResourceHandler("/formal/**").addResourceLocations("file:" + formalPath);
        registry.addResourceHandler(modelDescPath + "**").addResourceLocations("file:" + modelDescPath);
        registry.addResourceHandler(modelCasePath + "**").addResourceLocations("file:" + modelCasePath);
    }

    /*更改程序映射请求路径默认策略*/
    @Override
    public void configurePathMatch(PathMatchConfigurer configurer) {
        UrlPathHelper urlPathHelper = new UrlPathHelper();
        urlPathHelper.setUrlDecode(false);
        urlPathHelper.setDefaultEncoding(StandardCharsets.UTF_8.name());
        configurer.setUrlPathHelper(urlPathHelper);
    }


}
