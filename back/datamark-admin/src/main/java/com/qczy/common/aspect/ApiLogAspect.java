/*
package com.qczy.common.aspect;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.AfterReturning;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.aspectj.lang.annotation.Pointcut;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterNameDiscoverer;
import org.springframework.core.StandardReflectionParameterNameDiscoverer;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import javax.servlet.http.HttpServletRequest;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Aspect
@Component
public class ApiLogAspect {
    private final Logger logger = LoggerFactory.getLogger(this.getClass());
    private final ParameterNameDiscoverer parameterNameDiscoverer = new StandardReflectionParameterNameDiscoverer();

    @Pointcut("execution(* com.qczy.controller..*.*(..))")
    public void apiPointcut() {}

    @Before("apiPointcut()")
    public void before(JoinPoint joinPoint) {
        ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
        if (attributes != null) {
            HttpServletRequest request = attributes.getRequest();

            // 记录请求信息
            Map<String, Object> logInfo = new HashMap<>();
            logInfo.put("请求路径", request.getRequestURI());
            logInfo.put("请求方法", request.getMethod());
            logInfo.put("请求IP", request.getRemoteAddr());
            logInfo.put("请求参数", getRequestParams(joinPoint));

            logger.info("接口请求: {}", logInfo);
        }
    }

    // 其他方法保持不变...

    // 修改后的获取参数名方法
    private Map<String, Object> getRequestParams(JoinPoint joinPoint) {
        Map<String, Object> params = new HashMap<>();

        // 获取方法参数类型
        Class<?>[] paramTypes = ((org.aspectj.lang.reflect.MethodSignature) joinPoint.getSignature()).getParameterTypes();
        Object[] paramValues = joinPoint.getArgs();

        // 使用 StandardReflectionParameterNameDiscoverer 获取参数名
        String[] paramNames = parameterNameDiscoverer.getParameterNames(
                ((org.aspectj.lang.reflect.MethodSignature) joinPoint.getSignature()).getMethod()
        );

        if (paramNames != null) {
            for (int i = 0; i < paramNames.length; i++) {
                // 不记录文件内容，只记录文件名
                if (paramValues[i] instanceof org.springframework.web.multipart.MultipartFile) {
                    params.put(paramNames[i], ((org.springframework.web.multipart.MultipartFile) paramValues[i]).getOriginalFilename());
                } else {
                    params.put(paramNames[i], paramValues[i]);
                }
            }
        } else {
            // 如果无法获取参数名，使用参数类型和索引作为替代
            for (int i = 0; i < paramValues.length; i++) {
                params.put("arg" + i + "(" + paramTypes[i].getSimpleName() + ")",
                        paramValues[i] instanceof org.springframework.web.multipart.MultipartFile ?
                                ((org.springframework.web.multipart.MultipartFile) paramValues[i]).getOriginalFilename() :
                                paramValues[i]);
            }
        }

        return params;
    }
}*/
