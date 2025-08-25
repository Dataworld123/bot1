# bot1

    graph TB
    A[User Input] --> B[Intent Classifier Agent]
    B --> C{Query Type}
    C -->|Diagnosis| D[Diagnostic Agent]
    C -->|Treatment| E[Treatment Agent] 
    C -->|Prevention| F[Prevention Agent]
    C -->|General| G[General Consultation Agent]
    
    D --> H[Chain-of-Thought Processor]
    E --> H
    F --> H
    G --> H
    
    H --> I[Advanced RAG Pipeline]
    I --> J[Context Ranker]
    J --> K[Response Generator]
    K --> L[Quality Checker]
    L -->|Poor Quality| M[Reprompting System]
    M --> K
    L -->|Good Quality| N[Human-like Response Formatter]
    N --> O[Memory Manager]
    O --> P[Final Response]
    
    Q[AgentOps Monitor] --> R[Performance Analytics]
    Q --> S[Error Tracking]
    Q --> T[Response Optimization]
