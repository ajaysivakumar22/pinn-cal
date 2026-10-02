import React, { useState } from 'react';
import { CONCEPTS } from '../../data/concepts';
import { useAppStore } from '../../store/appStore';

export const ConceptModal: React.FC = () => {
  const { activeConcept, setActiveConcept } = useAppStore();
  const [disclosureLevel, setDisclosureLevel] = useState<'simple' | 'technical' | 'formula'>('simple');

  if (!activeConcept) return null;
  const concept = CONCEPTS[activeConcept];
  if (!concept) return null;

  return (
    <div
      className="modal-backdrop"
      onClick={() => setActiveConcept(null)}
      role="dialog"
      aria-modal="true"
      aria-labelledby="concept-title"
    >
      <div
        className="modal-content card"
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: 580,
          width: '90%',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
        }}
      >
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span
              style={{
                fontSize: '0.625rem',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                color: 'var(--color-primary)',
                fontWeight: 600,
              }}
            >
              {concept.category}
            </span>
            <h3 id="concept-title" style={{ fontSize: '1.05rem', margin: '2px 0 0 0', color: 'var(--color-text-primary)' }}>
              {concept.name}
            </h3>
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => setActiveConcept(null)}
            style={{ padding: '2px 8px', fontSize: '0.8125rem' }}
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Summary */}
          <div
            style={{
              padding: '8px 12px',
              background: '#f1f5f9',
              borderRadius: 6,
              fontSize: '0.8125rem',
              color: 'var(--color-text-secondary)',
              fontWeight: 500,
            }}
          >
            {concept.summary}
          </div>

          {/* Progressive disclosure selector */}
          <div style={{ display: 'flex', gap: 6, borderBottom: '1px solid var(--color-border)', paddingBottom: 8 }}>
            <button
              className={`mode-tab ${disclosureLevel === 'simple' ? 'active' : ''}`}
              onClick={() => setDisclosureLevel('simple')}
              style={{ fontSize: '0.75rem', padding: '4px 10px' }}
            >
              1. Plain Language
            </button>
            <button
              className={`mode-tab ${disclosureLevel === 'technical' ? 'active' : ''}`}
              onClick={() => setDisclosureLevel('technical')}
              style={{ fontSize: '0.75rem', padding: '4px 10px' }}
            >
              2. Technical Model
            </button>
            {concept.formula && (
              <button
                className={`mode-tab ${disclosureLevel === 'formula' ? 'active' : ''}`}
                onClick={() => setDisclosureLevel('formula')}
                style={{ fontSize: '0.75rem', padding: '4px 10px' }}
              >
                3. Mathematics
              </button>
            )}
          </div>

          {/* Level 1: Simple */}
          {disclosureLevel === 'simple' && (
            <div style={{ fontSize: '0.8125rem', lineHeight: 1.6, color: 'var(--color-text-primary)' }}>
              <p style={{ margin: 0 }}>{concept.simpleExplanation}</p>
            </div>
          )}

          {/* Level 2: Technical */}
          {disclosureLevel === 'technical' && (
            <div style={{ fontSize: '0.8125rem', lineHeight: 1.6, color: 'var(--color-text-primary)' }}>
              <p style={{ margin: 0 }}>{concept.technicalExplanation}</p>
            </div>
          )}

          {/* Level 3: Formula */}
          {disclosureLevel === 'formula' && concept.formula && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div
                style={{
                  padding: '12px 14px',
                  background: '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: 6,
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.8125rem',
                  color: '#0f172a',
                  overflowX: 'auto',
                }}
              >
                {concept.formula}
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-text-tertiary)', margin: 0 }}>
                Formal definition implemented in the frozen scientific core.
              </p>
            </div>
          )}

          {/* Why It Matters */}
          <div
            style={{
              padding: '10px 12px',
              background: '#fefce8',
              border: '1px solid #fef08a',
              borderRadius: 6,
              fontSize: '0.75rem',
              color: '#854d0e',
              lineHeight: 1.5,
            }}
          >
            <strong>Why This Matters in 5G:</strong> {concept.whyItMatters}
          </div>
        </div>
      </div>
    </div>
  );
};

export const ConceptTrigger: React.FC<{ conceptKey: string; label?: string }> = ({
  conceptKey,
  label = '?',
}) => {
  const { setActiveConcept } = useAppStore();

  return (
    <button
      type="button"
      className="concept-trigger-btn"
      onClick={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setActiveConcept(conceptKey);
      }}
      title={`Explain ${conceptKey.replace('_', ' ')}`}
      aria-label={`Learn about ${conceptKey.replace('_', ' ')}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: 16,
        height: 16,
        borderRadius: '50%',
        background: '#e0f2fe',
        color: '#0369a1',
        border: '1px solid #bae6fd',
        fontSize: '0.625rem',
        fontWeight: 700,
        cursor: 'pointer',
        marginLeft: 4,
        padding: 0,
        lineHeight: 1,
        verticalAlign: 'middle',
      }}
    >
      {label}
    </button>
  );
};
