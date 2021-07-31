

# -----------------------------------------------------------------------------------------------

# Aplicativo IWLS

# -----------------------------------------------------------------------------------------------

# Carrega Bibliotecas
library(shiny)
library(shinydashboard)
library(tidyverse)
library(tibble)
library(ggplot2)
library(gganimate)
library(hrbrthemes)
library(viridis)
library(gifski)
library(transformr)
library(metR)

# --------------------------------------------------------------------------

# Define funções

# --------------------------------------------------------------------------

# Calcula Y
calculaY <- function(eta){
  
  # Calcula Y Poisson
  exp(eta)
  
}

# Declara função calcula W de W
calcula_W <- function(eta, theta){
  
  # Calcula W Poisson
  (1 / theta) * exp(2 * eta)
  
}

# Declara função Z
calcula_Z <- function(eta, Y){
  
  # Calcula z Poisson
  eta + ( Y * exp(-eta) ) - 1 
  
}

#Declara funcao theta
calcula_theta <- function(eta, dist){
  
    # Calcula theta poisson
    exp(eta)
  
}

# Funcao estimativa de betas
EMV <- function(Y, Xs, beta, tol, norma){
  
  # Declara betas
  betas <- tibble(iter = 1,
                  beta0 = as.numeric(beta[1]),
                  beta1 = as.numeric(beta[2]))
  
  # Declara norma
  norma <- norma
  
  # Declara iteracao
  r = 1
  
  # Laco para popular betas calculados nas iterações
  while(norma >= tol){
    
    # Recupera beta da vez
    beta <- t(betas[r, c(2,3) ] %>% as.matrix())
    
    # Calcula eta0
    eta <- Xs %*% beta
    
    # Calcula media inicial
    theta <- calcula_theta(eta)
    
    # Calcula diagonal da matriz Wi
    W_i <- calcula_W(eta, theta)
    
    # Declara matriz W
    W <- diag(length(Y))
    
    # Popula diagonal da matriz W
    diag(W) <- W_i
    
    # Calcula z
    z_i <- calcula_Z(eta, Y)
    
    # Calcula beta_0
    betas[r+1, "iter" ] <- r + 1
    
    # Calcula beta_0
    betas[r+1, "beta0" ] <- (solve( t(Xs) %*% W %*% Xs ) %*% t(Xs) %*% W %*% z_i  )[1]
    
    # Calcula beta_1
    betas[r+1, "beta1" ] <- (solve( t(Xs) %*% W %*% Xs ) %*% t(Xs) %*% W %*% z_i  )[2]
    
    # Calcula norma euclideana
    norma <- sqrt(sum((abs(betas[r+1, c(2,3)] - betas[r, c(2,3)]))^2))
    
    # Incrementa r
    r = r + 1
    
  }
  
  # Retorno da função
  return(betas)
  
}

# Declara função calcula da devinace
calcula_deviance <- function(y, mu){
  
  # Retonra deviance
  return(round(2 * sum(y * log(y / mu) - (y - mu)), 6))
  
}


# --------------------------------------------------------------------------

# Interface do Usuário

# --------------------------------------------------------------------------

# Declara Interface do usuario
ui <- dashboardPage(
  
  # Cabecalho do aplicativo
  dashboardHeader(
    title = tags$img(src="imagens/optimizador_iwls4.png", width = '100%')
  ),
  
  # Declara barra lateral
  dashboardSidebar(
    
    # Seletores de entrada do usuário
    sliderInput("obs",
                "Número de Observações:",
                min = 10,
                max = 100,
                value = 25),
    sliderInput("beta0_real",
                "Beta 0 Real:",
                min = 0.1,
                max = 3,
                value = 2,
                step = 0.1),
    sliderInput("beta1_real",
                "Beta 1 Real",
                min = -5,
                max = -0.1,
                value = -2,
                step = 0.1),
    actionButton(inputId = "amostre", 
                 label = "Amostrar",
                 style="color: #fff; background-color: #f76f02; border-color: #2e6da4"),
    sliderInput("beta0",
                "Beta 0 Inicial",
                min = 0.01,
                max = 5,
                value = 1,
                step = 0.1),
    sliderInput("beta1",
                "Beta 1 Inicial",
                min = -5,
                max = -0.1,
                value = -11,
                step = 0.1),
    sliderInput("tol",
                "Tolerância",
                min = 0.01,
                max = 0.99,
                value = 0.1,
                step = 0.01),
    actionButton(inputId = "optimize", 
                 label = "Otimizar",
                 style="color: #fff; background-color: #f76f02; border-color: #2e6da4")
    
  ),
  
  # Declara corpo do aplicativo
  dashboardBody(
    HTML('<script> document.title = "Otimizador IWLS"; </script>'),
    # Define html style
    tags$style(".skin-blue .main-header .logo { background-color: #3e8fce; }
                .skin-blue .main-header .logo:hover { background-color: #3e8fce;}
                .skin-blue .main-header .navbar { background-color: #3e8fce;}"),
    fluidRow(
      box(width = 6, imageOutput("grafico1")),
      box(width = 6, imageOutput("grafico2"))
    ),
    fluidRow(
      box(width = 6, valueBoxOutput("dev", width = 12)),
      box(width = 6, valueBoxOutput("dist", width = 12))
    )
  )
  
)

# -----------------------------------------------------------------------------------------------

# Declara Servidor

# -----------------------------------------------------------------------------------------------

# Declara função servidor
server <- function(input, output, session){
  
  # Observa clique do botao amostrar
  dados <- eventReactive(input$amostre, {
    
    # Declara betas reais
    betas_reais <- c(input$beta0_real, input$beta1_real) 
    
    # Declara X
    X <- sort(rnorm(input$obs, 0, 0.5))
    
    # Declara matriz de Xs
    Xs <- cbind(rep(1, length(X)), X)
    
    # Calcula eta0
    eta <- Xs %*% betas_reais
    
    # Declara vetor Y
    Y = calculaY(eta)
    
    # Declara data frame
    amostra <- tibble(Y, X)
    
    # Retorno da função
    return(amostra)
    
  })
  
  # Observa clique do botao optimizar
  animacao1 <- eventReactive(input$optimize, {
    
    # Declara Y
    Y = dados()$Y
    
    # Declara X
    X = dados()$X
    
    # Declara matriz de Xs
    Xs = cbind(rep(1, length(X)), X)
    
    # Declara beta 0
    beta0 = c(input$beta0, input$beta1)
    
    # Executa EMV
    sol = EMV(Y, Xs, beta0, input$tol, 100) 
    
    # Define estados
    estados = paste0(rep(sol$iter, each = length(Y)),"    ",
                     "Beta 0:   ", rep(round(sol$beta0,4), each = length(Y)),"    ",
                     "Beta 1:   ", rep(round(sol$beta1,4), each = length(Y)))
    
    # Gera data frame
    df_anime <- tibble(iter = rep(sol$iter, each = length(Y)),
                       estados = factor(estados, levels = unique(estados)),
                       beta0 = rep(sol$beta0, each = length(Y)),
                       beta1 = rep(sol$beta1, each = length(Y)),
                       X = rep(X, times = nrow(sol)),
                       Observado = rep(Y, times = nrow(sol)),
                       pred = 0.1,
                       Y = "Estimado")
    
    # Laco para popular thetas
    for(i in 1:nrow(df_anime)){
      df_anime[i, "pred"] = exp(df_anime$beta0[i] + df_anime$beta1[i] * df_anime$X[i])  
    }
    
    # Data frame original
    original <- tibble(iter = rep(df_anime$iter[nrow(df_anime)], length(Y)),
                       estados = rep(df_anime$estados[nrow(df_anime)], length(Y)),
                       beta0 = rep(df_anime$beta0[nrow(df_anime)], length(Y)),
                       beta1 = rep(df_anime$beta1[nrow(df_anime)], length(Y)),
                       X = X,
                       Observado = Y,
                       pred = Y,
                       Y = "Observado")
    
    # Declara Data
    resultados <- df_anime %>% 
                    rbind(original)
    
    # Retorno
    return(resultados)
    
  })
  
  # calcula deviance
  deviance <- eventReactive(input$optimize,{
    
    # Declara Y
    Y = dados()$Y
    
    # Declara X
    X = dados()$X
    
    # Declara matriz de Xs
    Xs = cbind(rep(1, length(X)), X)
    
    # Declara beta 0
    beta0 = c(input$beta0, input$beta1)
    
    # Executa EMV
    sol = EMV(Y, Xs, beta0, input$tol, 100) 
    
    # Calcula mu
    mu <- exp(sol$beta0[nrow(sol)] + sol$beta1[nrow(sol)] * X)
    
    # Calcula deviance
    dev <- calcula_deviance(Y, mu)
    
    # Retorno da função
    return(dev)
    
  })
  
  # Calcula deviance 
  output$dev <- renderValueBox({
    
    # Condição de cor do termometro
    if(deviance() > 0.5){
      
      # Retorno da caixa de valor
      valueBox(
        "Deviance",
        deviance(),
        color = "red",
        icon = icon("thermometer-half"))
      
    } else {
      
      # Retorno da caixa de valor
      valueBox(
        "Deviance",
        deviance(),
        color = "green",
        icon = icon("thermometer-half"))

    }
    
  })
  
  # Renderiza Grafico
  output$grafico1 <- renderImage({
    
    # Declara animacao
    anime <- animacao1()
    
    # Declara objeto gráfico
    p <- anime %>%
      ggplot( aes(x=X, y=pred, group = Y, color = Y, shape = Y)) +
      geom_point(size = 5) +
      scale_shape_manual(values=c(1, 20))+
      scale_colour_manual(values = c("red", "blue"))+
      labs(title = "Curva de Ajuste da Distribuição Poisson",
           subtitle = paste("Iteração:  ","{closest_state}"),
           y = "Y")+
      transition_states(estados,
                        transition_length = 2,
                        state_length = 4,
                        wrap = FALSE) +
      view_follow(fixed_x = T)+
      theme(panel.grid = element_blank(),
            panel.background = element_blank(),
            text = element_text(size=13))
    
    # Salva animacao
    anim_save("outfile.gif", animate(p, fps = 5, renderer = gifski_renderer(loop = F))) 
    
    # Retorna lista contendo o arquivo
    list(src = "outfile.gif",
         contentType = 'image/gif',
         width = 550,
         height = 400)
    
  }, deleteFile = TRUE)
  
  # Observa clique do botao optimizar
  animacao2 <- eventReactive(input$optimize, {
    
    # Declara Y
    Y = dados()$Y
    
    # Declara X
    X = dados()$X
    
    # Declara matriz de Xs
    Xs = cbind(rep(1, length(X)), X)
    
    # Declara betas reais
    betas_reais <- c(input$beta0_real, input$beta1_real) 
    
    # Declara beta 0
    beta0 = c(input$beta0, input$beta1)
    
    # Executa EMV
    sol = EMV(Y, Xs, beta0, input$tol, 100) 
    
    # Declara betas 0 e betas 1
    betas0 <- seq(round(min(sol$beta0) -10),round(max(sol$beta0)+10),1)
    betas1 <- seq(round(min(sol$beta1) -10),round(max(sol$beta1)+10),1)
    
    # Define data frame espaço de busca
    espaco <- tibble(beta0 = rep(betas0, times = length(betas1)),
                     beta1 = rep(betas1, each = length(betas0))) %>% 
      mutate(dif_betas = sqrt((beta0 - betas_reais[1])^2 +
                                (beta1 - betas_reais[2])^2))
    
    # Retorno
    return(list(sol, espaco, betas_reais))
    
  })
  
  # Renderiza Grafico
  output$grafico2 <- renderImage({
    
    # Recupera objetos
    sol <- animacao2()[[1]]
    espaco <- animacao2()[[2]]
    
    # Declara grafico ggplot
    p <- ggplot(aes(x = beta0, y = beta1), data = sol)+
      geom_tile(show.legend = T,data = espaco, mapping = aes(x = beta0, y = beta1, fill = dif_betas) )+
      geom_contour(data = espaco, mapping = aes(x = beta0, y = beta1, z = dif_betas), color = "White" )+
      geom_text_contour(data = espaco,
                        aes(x = beta0, y = beta1, z = dif_betas),
                        color = "white",
                        size = 6,
                        min.size = 1,
                        skip = 0)+
      geom_point(color = "red", size = 4)+
      transition_time(iter) +
      view_follow(fixed_x = T, fixed_y = T)+
      labs(title = "Distância Euclidiana de Betas Reais e Estimados",
           x = "Beta 0",
           y = "Beta 1",
           fill = "Distância",
           subtitle = paste("Frame:  ","{frame_time}"))+
      theme(panel.grid = element_blank(),
            panel.background = element_blank(),
            text = element_text(size=13))
    
    # Salva animacao
    anim_save("outfile.gif", animate(p, fps = 5, renderer = gifski_renderer(loop = F))) 
    
    # Retorna lista contendo o arquivo
    list(src = "outfile.gif",
         contentType = 'image/gif',
         width = 550,
         height = 400)
    
  }, deleteFile = TRUE)
  
  # Calcula deviance 
  output$dist <- renderValueBox({
    
    # Recupera solucao
    sol = animacao2()[[1]]
    
    # Recupera Betas reais 
    betas_reais <- animacao2()[[3]]
    
    # Diferença de vetor de betas
    dif_betas <- round(sqrt((sol$beta0[nrow(sol)] - betas_reais[1] )^2 +
                            (sol$beta1[nrow(sol)] - betas_reais[2] )^2 ),
                       4)
    
    # Condição de cor do termometro
    if(dif_betas > 0.01){
      
      # Retorno da caixa de valor
      valueBox(
        "Distância",
        dif_betas,
        color = "red",
        icon = icon("drafting-compass"))
      
      
    } else {
      
      # Retorno da caixa de valor
      valueBox(
        "Distância",
        dif_betas,
        color = "green",
        icon = icon("drafting-compass"))
      
    }
    
  })
  
}

# -----------------------------------------------------------------------------------------------

# Executa função

# -----------------------------------------------------------------------------------------------


# Executa aplicativo
shinyApp(ui, server)

# -----------------------------------------------------------------------------------------------